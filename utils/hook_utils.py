def remove_all_hooks(module):
    """
    Recursively strip *every* forward / pre-forward / backward hook
    from `module` and its sub-modules.

    Use only for debugging; live code should track hook handles instead.

    ex. remove_all_hooks(model.policy)
    """
    for m in module.modules():          # walk the whole tree
        for attr in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"):
            d = getattr(m, attr, None)
            if d:                       # OrderedDict of {id: hook_fn}
                d.clear()

