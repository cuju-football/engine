# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""Builds static shell completions from the installed parsers."""

from __future__ import annotations

import argparse
import shlex

from ._cli_contract import subcommands

_FILES = {'FILE': 'file', 'DIR': 'directory', 'PATH': 'file'}


def _tree(root, packages):
    def walk(path, tree):
        children = (
            {name: child for group in subcommands(tree) for name, child in group.choices.items()}
            if tree is not None
            else {}
        )
        if not path:
            children.update(packages)
        yield path, tree, children
        for name, child in children.items():
            yield from walk((*path, name), child)

    return list(walk((), root))


def _values(action):
    if action.choices is not None:
        return tuple(map(str, action.choices)), ''
    return (), _FILES.get(action.metavar, '') if isinstance(action.metavar, str) else ''


def _actions(tree):
    if tree is None:
        return []
    return [
        action
        for action in tree._actions
        if action.help != argparse.SUPPRESS and not isinstance(action, argparse._SubParsersAction)
    ]


def _rows(nodes):
    for path, tree, children in nodes:
        name = ' '.join(path)
        flags, positionals, files = [], [], ''
        for action in _actions(tree):
            words, kind = _values(action)
            flags.extend(action.option_strings)
            if not action.option_strings:
                positionals.extend(words)
                files = files or kind
            elif action.nargs != 0:
                for option in action.option_strings:
                    yield f'{name}|{option}', words, kind
        yield name, (*children, *flags, *positionals), files


def _transitions(nodes):
    for path, tree, children in nodes:
        name = ' '.join(path)
        for child in children:
            yield f'{name}|{child}', ' '.join((*path, child))
        for action in _actions(tree):
            if action.nargs != 0:
                for flag in action.option_strings:
                    yield f'{name}|{flag}', action.nargs == '?'


def _bourne(nodes, shell):
    quote = shlex.quote
    body = [
        '_cuju_context() {',
        '  local context= pending= optional= word flag',
        '  for word in "$@"; do',
        '    if [ -n "$pending" ]; then',
        '      [ "$word" = "=" ] && continue',
        '      pending=',
        '      if [ -z "$optional" ] || [[ "$word" != -* ]]; then continue; fi',
        '    fi',
        '    flag=${word%%=*}',
        '    case "$context|$flag" in',
    ]
    for key, value in _transitions(nodes):
        if isinstance(value, str):
            code = f'context={quote(value)}'
        else:
            optional = 'yes' if value else "''"
            code = f'[ "$word" != "$flag" ] || {{ pending=$flag; optional={optional}; }}'
        body.append(f'      {quote(key)}) {code} ;;')
    body += [
        '    esac',
        '  done',
        '  if [ -n "$optional" ] && [[ "$current" == -* ]]; then pending=; fi',
        '  case "$context${pending:+|$pending}" in',
    ]
    for key, words, files in _rows(nodes):
        if shell == 'bash':
            command = ' '.join(['_cuju_offer', quote(files), *map(quote, words)])
        else:
            parts = []
            if words:
                parts.append('compadd -- ' + shlex.join(words))
            if files:
                parts.append('_files -/' if files == 'directory' else '_files')
            command = '; '.join(parts) or ':'
        body.append(f'    {quote(key)}) {command} ;;')
    body += ['  esac', '}']
    if shell == 'bash':
        body += [
            '_cuju_offer() {',
            '  local files="$1" candidate',
            '  shift',
            '  for candidate in "$@"; do',
            '    [[ "$candidate" == "$current"* ]] && COMPREPLY+=("$prefix$candidate")',
            '  done',
            '  if [[ -n "$files" ]]; then',
            '    [[ "$files" == directory ]] && files=-d || files=-f',
            '    while IFS= read -r candidate; do',
            '      COMPREPLY+=("$prefix$candidate")',
            '    done < <(compgen "$files" -- "$current")',
            '    compopt -o filenames',
            '  fi',
            '}',
            '_cuju() {',
            '  local current="${COMP_WORDS[COMP_CWORD]}" prefix=',
            '  local previous=("${COMP_WORDS[@]:1:COMP_CWORD-1}")',
            '  COMPREPLY=()',
            '  if [[ "$current" == -*=* ]]; then',
            '    prefix="${current%%=*}="',
            '    previous+=("${prefix%=}")',
            '    current="${current#*=}"',
            '  fi',
            '  _cuju_context "${previous[@]}"',
            '}',
            'complete -F _cuju cuju',
        ]
    else:
        body += [
            '_cuju_complete() {',
            '  local current="$PREFIX"',
            '  local -a previous',
            '  previous=("${(@)words[2,CURRENT-1]}")',
            '  if [[ "$current" == -*=* ]]; then',
            '    previous+=("${current%%=*}")',
            "    compset -P '*='",
            '    current="$PREFIX"',
            '  fi',
            '  _cuju_context "${previous[@]}"',
            '}',
            'if [[ "${funcstack[1]}" == _cuju ]]; then',
            '  _cuju_complete',
            'elif (( $+functions[compdef] )); then',
            '  compdef _cuju_complete cuju',
            'fi',
        ]
    return body


def _fish_quote(value):
    return "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"


def _fish(nodes):
    quote = _fish_quote
    body = [
        'function __cuju_context',
        '    set -l words (commandline -opc)',
        '    set -l context root',
        '    set -l pending 0',
        '    for word in $words[2..-1]',
        '        if test $pending -eq 1',
        '            set pending 0',
        '            continue',
        '        end',
        '        switch "$context|$word"',
    ]
    for key, value in _transitions(nodes):
        key = key if not key.startswith('|') else 'root' + key
        code = f'set context {quote(value)}' if isinstance(value, str) else 'set pending 1'
        body.extend((f'            case {quote(key)}', f'                {code}'))
    body += ['        end', '    end', '    echo "$context"', 'end', 'complete -c cuju -f']
    for path, tree, children in nodes:
        context = quote(' '.join(path) or 'root')
        command = 'complete -c cuju -n ' + quote(f'test (__cuju_context) = {context}')
        for child in children:
            body.append(f'{command} -a {quote(quote(child))}')
        for action in _actions(tree):
            args = []
            for flag in action.option_strings:
                prefix, value = ('-l', flag[2:]) if flag.startswith('--') else ('-s', flag[1:])
                args.extend((prefix, quote(value)))
            if action.option_strings and action.nargs not in (0, '?'):
                args.append('-r')
            values, files = _values(action)
            if values:
                args.extend(('-a', quote(' '.join(map(quote, values)))))
            if files == 'directory':
                args.extend(('-a', quote('(__fish_complete_directories (commandline -ct))')))
            elif files:
                args.append('-F')
            if args:
                body.append(f'{command} {" ".join(args)}')
    return body


def _powershell(nodes):
    def quote(value):
        return "'" + value.replace("'", "''") + "'"

    body = [
        'Register-ArgumentCompleter -Native -CommandName cuju -ScriptBlock {',
        '  param($wordToComplete, $commandAst, $cursorPosition)',
        "  $context = ''",
        "  $option = ''",
        '  foreach ($element in ($commandAst.CommandElements | Select-Object -Skip 1)) {',
        '    if ($element.Extent.EndOffset -ge $cursorPosition) { break }',
        "    if ($option) { $option = ''; continue }",
        '    $word = $element.Value',
        '    if ($null -eq $word) { $word = $element.Extent.Text }',
        '    switch -CaseSensitive -Exact ("$context|$word") {',
    ]
    for key, value in _transitions(nodes):
        code = f'$context = {quote(value)}' if isinstance(value, str) else '$option = $word'
        body.append(f'      {quote(key)} {{ {code}; break }}')
    body += [
        '    }',
        '  }',
        '  if ($option) { $context = "$context|$option" }',
        '  $candidates = @()',
        "  $files = ''",
        '  switch -CaseSensitive -Exact ($context) {',
    ]
    for key, words, files in _rows(nodes):
        candidates = ', '.join(map(quote, words))
        values = f'$candidates = @({candidates}); $files = {quote(files)}'
        body.append(f'    {quote(key)} {{ {values} }}')
    body += [
        '  }',
        '  foreach ($candidate in $candidates) {',
        '    if ($candidate.StartsWith($wordToComplete, [StringComparison]::OrdinalIgnoreCase)) {',
        '      $value = $candidate',
        '      $quote = [string][char]39',
        '      if ($value -match "[\\s$quote]") {',
        '        $value = $quote + $value.Replace($quote, $quote + $quote) + $quote',
        '      }',
        '      [System.Management.Automation.CompletionResult]::new(',
        '        $value, $candidate, "ParameterValue", $candidate)',
        '    }',
        '  }',
        '  if ($files) {',
        '    $completers = [System.Management.Automation.CompletionCompleters]',
        '    $completers::CompleteFilename($wordToComplete) |',
        "      Where-Object { $files -eq 'file' -or $_.ResultType -eq 'ProviderContainer' }",
        '  }',
        '}',
    ]
    return body


def script(shell, root, packages, versions):
    """Returns the completion script for one shell."""
    nodes = _tree(root, {name: tree for name, _, tree in packages})
    if shell in ('bash', 'zsh'):
        body = _bourne(nodes, shell)
    elif shell == 'fish':
        body = _fish(nodes)
    else:
        body = _powershell(nodes)
    header = '# Cuju completion for ' + ', '.join(' '.join(line.splitlines()) for line in versions)
    lines = ['#compdef cuju', header] if shell == 'zsh' else [header]
    return '\n'.join([*lines, *body])
