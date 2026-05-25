import argparse

def add_minus_flags(parser: argparse.ArgumentParser):
    """
    Automatically add aliases for argparse options that contain underscores.
    For example:
    `--fix_existing` -> add alias `--fix-existing`
    `--augment_OD_num` -> add alias `--augment-OD-num`
    """
    for action in parser._actions:
        # Only process optional arguments, i.e. those starting with `-`.
        if not action.option_strings:
            continue
        aliases_to_add = []
        for opt in action.option_strings:
            if opt.startswith('--') and '_' in opt:
                new_alias = opt.replace('_', '-')
                if new_alias not in action.option_strings:
                    aliases_to_add.append(new_alias)
        for alias in aliases_to_add:
            action.option_strings.append(alias)
            parser._option_string_actions[alias] = action
    return parser


def add_negation_flags(parser: argparse.ArgumentParser):
    """
    Automatically add matching `--no-xxx` options for `store_true` flags.
    """
    args_to_add = []
    
    # Build an index of existing flags to avoid duplicates.
    existing_flags = set()
    for action in parser._actions:
        existing_flags.update(action.option_strings)
    for action in parser._actions:
        if isinstance(action, argparse._StoreTrueAction):
            # Collect all possible negated aliases for the current action.
            negation_aliases = []
            for option in action.option_strings:
                if not option.startswith('--'): 
                    continue
                # Generate the negated form: `--use_new_model` -> `--no-use_new_model`
                raw_name = option.removeprefix('--')
                neg_opt = f'--no-{raw_name}'
                # Add it only if the flag does not already exist.
                if neg_opt not in existing_flags:
                    negation_aliases.append(neg_opt)
            # If valid negated aliases were generated, queue them for addition.
            if negation_aliases:
                # Use the first option name as the help display name.
                primary_name = action.option_strings[0].removeprefix('--')
                args_to_add.append({
                    'options': negation_aliases,  # This is a list.
                    'dest': action.dest,
                    'action': 'store_false',
                    'default': action.default,
                    'help': f"Disable {primary_name}"
                })
    for arg in args_to_add:
        parser.add_argument(
            *arg['options'], 
            dest=arg['dest'],
            action=arg['action'],
            default=arg['default'],
            help=arg['help']
        )
    return parser
