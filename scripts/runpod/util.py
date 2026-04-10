#!/usr/bin/env python3
import json
import shlex
import sys
from pathlib import Path


STATUS_ORDER = {
    "RUNNING": 0,
    "CREATED": 1,
    "STARTING": 2,
    "PROVISIONING": 3,
    "PENDING": 4,
    "EXITED": 5,
    "STOPPED": 6,
    "FAILED": 7,
    "TERMINATED": 8,
}

SSH_OPTS_WITH_ARG = {
    "-b",
    "-c",
    "-D",
    "-E",
    "-e",
    "-F",
    "-I",
    "-i",
    "-J",
    "-L",
    "-l",
    "-m",
    "-O",
    "-o",
    "-p",
    "-Q",
    "-R",
    "-S",
    "-W",
    "-w",
}


def die(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(1)


def read_stdin() -> str:
    return sys.stdin.read()


def load_json_from_stdin():
    raw = read_stdin().strip()
    if not raw:
        return None
    return json.loads(raw)


def walk(value):
    if isinstance(value, dict):
      yield value
      for child in value.values():
        yield from walk(child)
    elif isinstance(value, list):
      for child in value:
        yield from walk(child)


def walk_strings(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_strings(child)
    elif isinstance(value, str):
        yield value


def first_key(value, keys):
    if isinstance(value, dict):
        for key in keys:
            if key in value:
                return value[key]
        for child in value.values():
            found = first_key(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_key(child, keys)
            if found is not None:
                return found
    return None


def find_named(items, name):
    if not isinstance(items, list):
        return None
    matches = [item for item in items if isinstance(item, dict) and item.get("name") == name]
    if not matches:
        return None
    matches.sort(key=lambda item: STATUS_ORDER.get(str(item.get("status", "")).upper(), 99))
    return matches[0]


def datacenter_for_gpu(data, gpu_id, preferred_ids, preferred_location):
    if not isinstance(data, list):
        return None

    def has_gpu(entry):
        for gpu in entry.get("gpuAvailability", []) or []:
            if gpu.get("gpuId") == gpu_id:
                return True
        return False

    candidates = [entry for entry in data if isinstance(entry, dict) and has_gpu(entry)]
    if not candidates:
        return None

    for wanted in preferred_ids:
        for entry in candidates:
            if entry.get("id") == wanted:
                return entry

    if preferred_location:
        for entry in candidates:
            if entry.get("location") == preferred_location:
                return entry

    return candidates[0]


def find_ssh_command(raw: str):
    raw = raw.strip()
    if not raw:
        return None

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        for candidate in walk_strings(parsed):
            stripped = candidate.strip()
            if stripped.startswith("ssh "):
                return stripped

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("ssh "):
            return stripped

    if "ssh " in raw:
        tail = raw[raw.index("ssh ") :].strip()
        return tail.splitlines()[0].strip()
    return None


def parse_ssh_command(command: str):
    tokens = shlex.split(command)
    if not tokens or tokens[0] != "ssh":
        die(f"Unsupported ssh command: {command}")

    args = []
    target = None
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token == "--":
            break
        if token.startswith("-"):
            args.append(token)
            if token in SSH_OPTS_WITH_ARG:
                i += 1
                if i >= len(tokens):
                    die(f"Missing argument for ssh option {token}")
                args.append(tokens[i])
            i += 1
            continue
        if target is None:
            target = token
            i += 1
            continue
        break

    if target is None:
        die(f"Could not find ssh target in: {command}")

    ssh_prefix = ["ssh", *args, target]
    rsync_shell = shlex.join(["ssh", *args])
    return {
        "target": target,
        "command": shlex.join(ssh_prefix),
        "rsync_shell": rsync_shell,
        "ssh_args": args,
        "ssh_prefix": ssh_prefix,
    }


def bash_assign_array(name: str, values):
    quoted = " ".join(shlex.quote(value) for value in values)
    return f"{name}=({quoted})"


def state_get(path_str: str, key: str):
    path = Path(path_str)
    if not path.exists():
        return ""
    data = json.loads(path.read_text())
    value = data.get(key, "")
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def state_set(path_str: str, key: str, value: str):
    path = Path(path_str)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
    data[key] = value
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    if len(sys.argv) < 2:
        die("usage: util.py <command> [args]")

    command = sys.argv[1]

    if command == "field":
        if len(sys.argv) < 3:
            die("usage: util.py field <key> [key...]")
        data = load_json_from_stdin()
        found = first_key(data, sys.argv[2:])
        if found is None:
            raise SystemExit(1)
        if isinstance(found, (dict, list)):
            print(json.dumps(found))
        else:
            print(found)
        return

    if command == "pod-from-list":
        if len(sys.argv) != 3:
            die("usage: util.py pod-from-list <name>")
        data = load_json_from_stdin()
        found = find_named(data, sys.argv[2])
        if found is None:
            raise SystemExit(1)
        print(json.dumps(found))
        return

    if command == "volume-from-list":
        if len(sys.argv) != 3:
            die("usage: util.py volume-from-list <name>")
        data = load_json_from_stdin()
        found = find_named(data, sys.argv[2])
        if found is None:
            raise SystemExit(1)
        print(json.dumps(found))
        return

    if command == "datacenter-for-gpu":
        if len(sys.argv) != 5:
            die("usage: util.py datacenter-for-gpu <gpu-id> <preferred-ids-csv> <preferred-location>")
        data = load_json_from_stdin()
        preferred_ids = [item for item in sys.argv[3].split(",") if item]
        found = datacenter_for_gpu(data, sys.argv[2], preferred_ids, sys.argv[4])
        if found is None:
            raise SystemExit(1)
        print(json.dumps(found))
        return

    if command == "ssh-export":
        raw = read_stdin()
        ssh_command = find_ssh_command(raw)
        if ssh_command is None:
            die("Could not extract an ssh command from runpodctl output")
        parsed = parse_ssh_command(ssh_command)
        print(f"RUNPOD_SSH_TARGET={shlex.quote(parsed['target'])}")
        print(f"RUNPOD_SSH_COMMAND={shlex.quote(parsed['command'])}")
        print(f"RUNPOD_RSYNC_SHELL={shlex.quote(parsed['rsync_shell'])}")
        print(bash_assign_array("RUNPOD_SSH_ARGS", parsed["ssh_args"]))
        print(bash_assign_array("RUNPOD_SSH_PREFIX", parsed["ssh_prefix"]))
        return

    if command == "state-get":
        if len(sys.argv) != 4:
            die("usage: util.py state-get <path> <key>")
        print(state_get(sys.argv[2], sys.argv[3]))
        return

    if command == "state-set":
        if len(sys.argv) != 5:
            die("usage: util.py state-set <path> <key> <value>")
        state_set(sys.argv[2], sys.argv[3], sys.argv[4])
        return

    die(f"unknown command: {command}")


if __name__ == "__main__":
    main()
