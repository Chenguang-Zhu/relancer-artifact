

# entrance_2
def runBaseLineMigration(project, notebook, api_mapping, new_file):
    for old_api in api_mapping:
        new_api_candidates = api_mapping[old_api]
        migrateOneAPI_Baseline(old_api, new_api_candidates, project, notebook, new_file)


# helper_2
def migrateOneAPI_Baseline(old_api, new_api_candidates, project, notebook, new_file):
    cand = new_api_candidates[0]
    migrateOneAPIOneCand_Baseline(old_api, cand, project, notebook, new_file)


# helper_2
def migrateOneAPIOneCand_Baseline(old_api, new_api, project, notebook, new_file):
    with open(new_file, 'r') as fr:
        lines = fr.readlines()
    new_lines = []
    edited = False
    # edit lines
    for i, l in enumerate(lines):
        if l.strip().startswith('#'):
            continue
        suf = findLongestOldAPISuffixInLine(l, old_api)
        if suf is None:
            continue
        print ('SUFFIX ' + suf + ' MATCH ' + old_api)
        edited = True
        if l.strip().startswith('from ') or l.strip().startswith('import '):
            lines[i] = '\n'
        else:
            lines[i] = lines[i].replace(suf, new_api)
    if not edited:
        return
    # if edited any line, import new api's top level package
    top_level_pacakge_name = new_api.split('.')[0]
    already_imported = False
    for i, l in enumerate(lines):
        if 'import ' in l:
            if l.strip().split('import ')[1] == top_level_pacakge_name:
                already_imported = True
    if not already_imported:
        new_lines.append('import ' + top_level_pacakge_name + '\n')
    new_lines += lines
    with open(new_file, 'w') as fw:
        fw.write(''.join(new_lines))


# helper_2
def findLongestOldAPISuffixInLine(line, old_api):
    segs = old_api.split('.')
    suf = None
    while segs:
        s = '.'.join(segs)
        if s in line:
            suf = s
            break
        else:
            segs = segs[1:]
    return suf
