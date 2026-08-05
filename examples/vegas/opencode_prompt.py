#!/usr/bin/env python3

import argparse
import json
import sqlite3
from pathlib import Path


DEFAULT_DATABASE = Path.home() / ".local/share/opencode/opencode.db"
DEFAULT_OUTPUT = Path("/tmp/opencode-vegas-technical-conversations.json")

# Long, coherent technical conversations selected from the local OpenCode
# database. Keep the tutorial last so the transcript can end at its final user
# question and provide a natural continuation point for generation.
DEFAULT_SESSIONS = (
    "ses_2452593b5ffe6yJELmXEV6x9oM",  # Rebuilding MCP memory server from scratch
    "ses_1af9879efffe0sg7s2N3kxgaDo",  # Difference between resource and object
)


def message_text(connection, message_id):
    chunks = []
    for (data,) in connection.execute(
        "SELECT data FROM part WHERE message_id = ? ORDER BY time_created, id",
        (message_id,),
    ):
        part = json.loads(data)
        if part.get("type") == "text" and part.get("text", "").strip():
            chunks.append(part["text"].strip())
    return "\n\n".join(chunks)


def load_conversation(connection, session_id):
    session = connection.execute(
        "SELECT title, parent_id FROM session WHERE id = ?",
        (session_id,),
    ).fetchone()
    if session is None:
        raise RuntimeError(f"OpenCode session not found: {session_id}")
    title, parent_id = session
    if parent_id is not None:
        raise RuntimeError(f"OpenCode session is a child/background session: {session_id}")

    messages = []
    for message_id, data in connection.execute(
        "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created, id",
        (session_id,),
    ):
        metadata = json.loads(data)
        role = metadata.get("role")
        if role not in ("user", "assistant"):
            continue
        # Tool-call messages are usually short progress narration. Retain only
        # completed assistant replies so the fixture reads like a conversation.
        if role == "assistant" and metadata.get("finish") != "stop":
            continue
        text = message_text(connection, message_id)
        if text:
            messages.append({"role": role, "content": text})
    merged = []
    for message in messages:
        if merged and merged[-1]["role"] == message["role"]:
            merged[-1]["content"] += "\n\n" + message["content"]
        else:
            merged.append(message)
    return title, merged


def build_fixture(database, session_ids=DEFAULT_SESSIONS):
    uri = f"file:{database}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        conversations = [load_conversation(connection, session_id) for session_id in session_ids]

    fixture_messages = []
    sources = []
    reference = None
    for index, (title, session_messages) in enumerate(conversations, start=1):
        if not session_messages:
            raise RuntimeError(f"OpenCode session has no usable conversation text: {session_ids[index - 1]}")
        session_id = session_ids[index - 1]
        sources.append({"session_id": session_id, "title": title})
        for message in session_messages:
            message["source_session_id"] = session_id
            message["source_title"] = title

        if index == len(conversations):
            if session_messages[-1]["role"] != "assistant":
                raise RuntimeError(f"last OpenCode session does not end in an assistant reference: {session_id}")
            reference = session_messages.pop()
        fixture_messages.extend(session_messages)

    if reference is None or not fixture_messages or fixture_messages[-1]["role"] != "user":
        raise RuntimeError("fixture must end at the user message immediately before the reference answer")

    return {
        "schema_version": 1,
        "source": "opencode",
        "sessions": sources,
        "messages": fixture_messages,
        "reference": reference,
    }


def ensure_prompt(database=DEFAULT_DATABASE, output=DEFAULT_OUTPUT):
    fixture = build_fixture(database)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = ensure_prompt(args.database, args.output)
    print(f"wrote {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
