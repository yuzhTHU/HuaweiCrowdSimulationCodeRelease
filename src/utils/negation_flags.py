import argparse

def add_negation_flags(parser: argparse.ArgumentParser):
    """
    自动为 parser 中的 store_true 参数添加对应的 --no-xxx 选项。
    """
    for action in parser._actions:
        # 只处理布尔型的 store_true
        if isinstance(action, argparse._StoreTrueAction):
            # 获取参数名，例如 '--flag'
            for option in action.option_strings:
                if not option.startswith('--'): 
                    continue
                neg_option = '--no-' + option.removeprefix('--')
                if any(neg_option in a.option_strings for a in parser._actions):
                    continue # 避免重复添加
                parser.add_argument(
                    neg_option,
                    dest=action.dest,
                    action='store_false',
                    default=action.default,
                    help=f"Disable {option.removeprefix('--')}"
                )
    return parser
