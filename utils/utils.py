def make_dict_from_locals(locals, types=[], keys=[]):
    if len(keys)>0:
        if len(types)==0 :
            D= dict((k, locals[k]) for k in keys)
        else:
            D = dict((k, locals[k]) for k in keys if type(locals[k]) in types)
    else:
        if len(types)>0:
            D = dict((k, locals[k]) for k in locals.keys() if type(locals[k]) in types)
        else:
            D={}
    return D
