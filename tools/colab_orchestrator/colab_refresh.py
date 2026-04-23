#!/usr/bin/env python3
"""Runtime-token maintenance for colab-cli sessions. Run with the CLI's own interpreter.

The stock CLI stores the runtime-proxy token issued at `colab new` and never
renews it. It expires (tokenExpiresInSeconds, ~1 h); every later call 404s,
the CLI treats that as a dead VM, deletes the session and kills its
keep-alive, and the still-billing VM is reclaimed minutes later.
list_assignments() returns a current token for every assignment; these
subcommands use it through the CLI's own state API. Tokens are never printed.

  show NAME               ENDPOINT=...  TTL=...
  refresh NAME            swap the current token into the stored session
  adopt NAME ENDPOINT     re-create a pruned session for a still-assigned VM,
                          with a fresh token and a new keep-alive daemon
  orphans                 assigned endpoints with no local session
  unassign ENDPOINT       release an assignment by endpoint (billing stops)
  selftest                import check, no network

exit: 0 ok | 2 usage | 3 endpoint not listed (gone) | 4 no such local session
"""
import sys

from colab_cli.common import state


def _listed(endpoint):
    return next((a for a in state.client.list_assignments() if a.endpoint == endpoint), None)


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "selftest":
        from colab_cli.client import Variant  # noqa: F401
        from colab_cli.commands.session import SessionState, spawn_keep_alive  # noqa: F401
        print(f"SELFTEST_OK local_sessions={sorted(state.store.list())}")
        return 0
    if cmd == "show":
        s = state.store.get(args[0])
        if s is None:
            return 4
        print(f"ENDPOINT={s.endpoint}")
        a = _listed(s.endpoint)
        if a is None:
            return 3
        print(f"TTL={a.runtime_proxy_info.token_expires_in_seconds}")
        return 0
    if cmd == "refresh":
        s = state.store.get(args[0])
        if s is None:
            return 4
        a = _listed(s.endpoint)
        if a is None:
            return 3
        info = a.runtime_proxy_info
        changed = info.token != s.token
        s.token, s.url = info.token, info.url
        state.store.add(s)
        print(f"REFRESH_OK ttl={info.token_expires_in_seconds} changed={changed}")
        return 0
    if cmd == "adopt":
        name, endpoint = args
        a = _listed(endpoint)
        if a is None:
            return 3
        from colab_cli.client import Variant
        from colab_cli.commands.session import SessionState, spawn_keep_alive
        info = a.runtime_proxy_info
        variant = Variant[a.variant.name] if a.variant.name in Variant.__members__ else Variant.GPU
        s = SessionState(name=name, token=info.token, url=info.url, endpoint=endpoint,
                         variant=variant.value, accelerator=a.accelerator.value)
        state.store.add(s)
        s.keep_alive_pid = spawn_keep_alive(endpoint, name, auth_provider=state.auth_provider,
                                            config_path=state.config_path)
        state.store.add(s)
        print(f"ADOPTED ttl={info.token_expires_in_seconds} keepalive_pid={s.keep_alive_pid}")
        return 0
    if cmd == "orphans":
        local = {s.endpoint for s in state.store.list().values()}
        for a in state.client.list_assignments():
            if a.endpoint not in local:
                print(f"ORPHAN={a.endpoint} ACC={a.accelerator.value}")
        return 0
    if cmd == "unassign":
        if _listed(args[0]) is None:
            return 3
        state.client.unassign(args[0])
        print("UNASSIGNED")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
