#!/usr/bin/env python3
"""Exercise a real wyn-analyzer process with package-aware LSP requests.

Run: python3 scripts/check_analyzer_navigation.py target/debug/wyn-analyzer
Uses only the Python standard library and creates fixtures in a temporary folder.
"""
import json
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading


class Client:
    def __init__(self, binary, stderr):
        self.process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        self.messages = queue.Queue()
        self.sequence = 0
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        while True:
            headers = {}
            while True:
                line = self.process.stdout.readline()
                if not line:
                    return
                if line == b'\r\n':
                    break
                key, value = line.decode().split(':', 1)
                headers[key.lower()] = value.strip()
            self.messages.put(json.loads(self.process.stdout.read(int(headers['content-length']))))

    def send(self, method, params, request_id=None):
        message = {'jsonrpc': '2.0', 'method': method}
        if params is not None:
            message['params'] = params
        if request_id is not None:
            message['id'] = request_id
        body = json.dumps(message).encode()
        self.process.stdin.write(f'Content-Length: {len(body)}\r\n\r\n'.encode() + body)
        self.process.stdin.flush()

    def receive(self, predicate):
        while True:
            message = self.messages.get(timeout=60)
            if predicate(message):
                return message

    def request(self, method, params):
        self.sequence += 1
        self.send(method, params, self.sequence)
        result = self.receive(lambda message: message.get('id') == self.sequence)
        assert 'error' not in result, result
        return result.get('result')

    def open(self, path, source):
        self.send('textDocument/didOpen', {'textDocument': {
            'uri': path.as_uri(), 'languageId': 'wyn', 'version': 1, 'text': source}})
        return self.diagnostics(path)

    def change(self, path, source, version):
        self.send('textDocument/didChange', {'textDocument': {'uri': path.as_uri(), 'version': version},
                                           'contentChanges': [{'text': source}]})
        return self.diagnostics(path)

    def diagnostics(self, path):
        return self.receive(lambda message: message.get('method') == 'textDocument/publishDiagnostics'
                            and message['params']['uri'] == path.as_uri())['params']['diagnostics']

    def at(self, method, path, source, needle, **extra):
        offset = source.index(needle)
        prefix = source[:offset]
        position = {'line': prefix.count('\n'),
                    'character': len(prefix.rsplit('\n', 1)[-1].encode('utf-16-le')) // 2}
        return self.request('textDocument/' + method, {'textDocument': {'uri': path.as_uri()},
                                                     'position': position, **extra})

    def close(self):
        try:
            self.request('shutdown', None)
            self.send('exit', None)
            self.process.stdin.close()
            self.process.wait(timeout=5)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()


def run(binary):
    with tempfile.TemporaryDirectory(prefix='wyn-navigation-') as temporary:
        root = pathlib.Path(temporary).resolve()
        for name in ['app', 'math']:
            directory = root / name
            (directory / 'src').mkdir(parents=True)
            manifest = f'manifest-version = 1\n[package]\nname = "manual/{name}"\nversion = "v1.0.0"\nwyn = "v0.1.0"\nlibrary = "src/lib.wyn"\n'
            if name == 'app':
                manifest += '[dependencies]\nmath = { package = "manual/math", version = "v1.0.0", path = "../math" }\n'
            (directory / 'wyn.toml').write_text(manifest)
        library = root / 'math/src/lib.wyn'
        library_source = 'def double(value: i32) i32 = value + value\n'
        library.write_text(library_source)
        app = root / 'app/src/lib.wyn'
        source = 'module Math = import "pkg:math"\nentry main(value: i32) i32 = Math.double(value)\n'
        app.write_text(source)
        unopened = root / 'app/src/other.wyn'
        unopened.write_text('module Other = import "pkg:math"\ndef caller(value: i32) i32 = Other.double(value)\n')
        with (root / 'stderr.log').open('w+') as stderr:
            client = Client(binary, stderr)
            try:
                client.request('initialize', {'processId': None, 'rootUri': (root / 'app').as_uri(), 'capabilities': {}})
                client.send('initialized', {})
                assert client.open(app, source) == []
                definition = client.at('definition', app, source, 'double(value)')
                assert definition['uri'] == library.as_uri(), definition
                assert definition['range'] == {'start': {'line': 0, 'character': 4}, 'end': {'line': 0, 'character': 10}}, definition
                print('PASS: package definition jumps to exact name in dependency file', flush=True)
                references = client.at('references', app, source, 'double(value)', context={'includeDeclaration': False})
                assert sorted(item['uri'] for item in references) == sorted([app.as_uri(), unopened.as_uri()]), references
                declared = client.at('references', app, source, 'double(value)', context={'includeDeclaration': True})
                assert len(declared) == 3 and any(item['uri'] == library.as_uri() for item in declared), declared
                print('PASS: references include unopened caller with a different import alias; declaration flag works', flush=True)
                signature = client.at('signatureHelp', app, source, 'value)\n')
                assert 'i32' in signature['signatures'][0]['label'], signature
                assert client.at('hover', app, source, 'double(value)') is not None
                print('PASS: imported function hover and signature help', flush=True)
                incomplete = source.replace('Math.double(value)', 'Math.')
                assert client.change(app, incomplete, 2)
                completion = client.request('textDocument/completion', {'textDocument': {'uri': app.as_uri()},
                    'position': {'line': 1, 'character': len(incomplete.splitlines()[1])},
                    'context': {'triggerKind': 2, 'triggerCharacter': '.'}})
                assert any(item['label'] == 'double' for item in completion), completion
                assert client.change(app, source, 3) == []
                print('PASS: package member completion during incomplete edit', flush=True)
                moved = '-- unsaved dependency\n' + library_source
                assert client.open(library, moved) == []
                definition = client.at('definition', app, source, 'double(value)')
                assert definition['range']['start']['line'] == 1, definition
                reverse = client.at('references', library, moved, 'double(value:', context={'includeDeclaration': False})
                assert sorted(item['uri'] for item in reverse) == sorted([app.as_uri(), unopened.as_uri()]), reverse
                print('PASS: unsaved dependency positions and reverse references from its declaration', flush=True)
                incompatible = moved.replace('i32', 'bool').replace('value + value', 'value')
                assert client.change(library, incompatible, 2) == []
                assert client.diagnostics(app), 'open caller should report changed dependency type'
                client.send('textDocument/didClose', {'textDocument': {'uri': library.as_uri()}})
                assert client.diagnostics(app) == []
                print('PASS: dependency edit refreshes caller diagnostics; close restores disk source', flush=True)
                local = root / 'app/src/local.wyn'
                local_source = 'def first(value: i32) i32 = value\ndef second(value: i32) i32 = first(value)\n'
                local.write_text(local_source)
                assert client.open(local, local_source) == []
                refs = client.at('references', local, local_source, 'value: i32', context={'includeDeclaration': False})
                assert len(refs) == 1 and refs[0]['range']['start']['line'] == 0, refs
                definition = client.at('definition', local, local_source, 'first(value)')
                assert definition['range']['start'] == {'line': 0, 'character': 4}, definition
                print('PASS: local functions navigate; same-named parameters stay separate', flush=True)
                constants = 'def amount: i32 = 42\nentry main(value: i32) i32 = value + amount + amount\n'
                assert client.change(local, constants, 2) == []
                references = client.at('references', local, constants, 'amount:', context={'includeDeclaration': False})
                assert len(references) == 2, references
                definition = client.at('definition', local, constants, 'amount + amount')
                assert definition['range']['start'] == {'line': 0, 'character': 4}, definition
                print('PASS: folded integer constants retain definition and all references', flush=True)
            except BaseException:
                stderr.flush()
                stderr.seek(0)
                print(stderr.read(), file=sys.stderr)
                raise
            finally:
                client.close()


if __name__ == '__main__':
    run(str(pathlib.Path(sys.argv[1]).resolve()))
