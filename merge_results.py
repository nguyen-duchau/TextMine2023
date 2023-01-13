import json

CLASSES_REGEX = ['Email', 'Phone_Number', 'Reference_CEDEX', 'Reference_CS', 'Reference_Code_Postal', 'Social_Network',
                 'Url', 'Reference_User']

REGEX_DATA = 'annotations/preds_regex.json'
CRF_DATA = 'annotations/preds_crf.json'
BONZAI_DATA = 'annotations/preds_bonzai.json'
BERT_DATA = 'annotations/preds_bert.json'

if __name__ == '__main__':
    # print("Loading Predictions")
    with open(REGEX_DATA) as in_f:
        regex_data = json.load(in_f)
    with open(CRF_DATA) as in_f:
        crf_data = json.load(in_f)
    with open(BONZAI_DATA) as in_f:
        bonzai_data = json.load(in_f)
    with open(BERT_DATA) as in_f:
        bert_data = json.load(in_f)

    conflicts = 0
    votes_nocrf = 0
    votes_nobonzai = 0
    votes_nobert = 0
    conflicts_heuristic = 0

    full_list = []
    for reg_i, crf_i, bonzai_i, bert_i in zip(regex_data, crf_data, bonzai_data, bert_data):
        res = dict()
        res['identifier'] = bonzai_i['identifier']

        # print(reg_i['text'].replace('\n', ' \\n '))
        # print(crf_i['text'].replace('\n', ' \\n '))
        # print(bonzai_i['text'].replace('\n', ' \\n '))
        res['text'] = bonzai_i['text']

        res['annotations'] = []

        for reg_j, crf_j, bonzai_j, bert_j in zip(reg_i["annotations"],
                                                  crf_i["annotations"],
                                                  bonzai_i["annotations"],
                                                  bert_i["annotations"]):
            token = dict()
            token['form'] = bonzai_j['form']
            token['begin'] = bonzai_j['begin']
            token['end'] = bonzai_j['end']

            if reg_j["label"] in CLASSES_REGEX:
                token['label'] = reg_j["label"]
            else:
                if len(res['text'].split('\n')) == 1 and bert_j['label_bert'] != 'O': # Si le texte ne contient pas de retour à la ligne, laisser BERT prendre la décision (excluant 'O')
                    token['label'] = bert_j['label_bert']
                if crf_j['label'] != bonzai_j['label_bonzaiboost'] or crf_j['form'] != bert_j['label_bert'] or bonzai_j['label_bonzaiboost'] != bert_j.get('label_bert', 'O'):
                    # Compte le nombre de fois où au moins un des réseaux n'est pas d'accord avec les autres
                    conflicts += 1


                if crf_j['label'] == bonzai_j['label_bonzaiboost']:  # Si CRF et BonzaiBoost votent pour la même classe, prendre celle-là
                    # Contient également tous les cas où les trois sont d'accord
                    token['label'] = crf_j['label']
                    if crf_j['label'] != bert_j['label_bert'] and bert_j['label_bert'] != 'O':
                        # Compte le nombre de fois où CRF et BonzaiBoost étaient d'accord contre l'avis de BERT, en dehors de 'O'
                        votes_nobert += 1
                elif crf_j['label'] == bert_j['label_bert']: # Si CRF et BERT votent pour la même classe, prendre celle-là
                    token['label'] = crf_j['label']
                    if crf_j['label'] != bonzai_j['label_bonzaiboost']:
                        # Compte le nombre de fois où CRF et BERT étaient d'accord contre l'avis de Bonzaiboost
                        votes_nobonzai += 1
                elif bonzai_j['label_bonzaiboost'] == bert_j['label_bert']: # Si BonzaiBoost et BERT votent pour la même classe, prendre celle-là
                    token['label'] = bonzai_j['label_bonzaiboost']
                    if crf_j['label'] != bonzai_j['label_bonzaiboost'] and crf_j['label'] != 'None':
                        # Compte le nombre de fois où BERT et BonzaiBoost étaient d'accord contre l'avis de CRF, en dehors de 'None'
                        votes_nocrf += 1
                else:
                    conflicts_heuristic += 1
                    print(res['text'].replace('\n', '\\n'))
                    if crf_j['form'] != bonzai_j['form'] or crf_j['form'] != bert_j['form']:
                        print(crf_j['form'], bonzai_j['form'], bert_j['form'])
                        raise Exception
                    print(crf_j['form'])
                    print(crf_j['label'], bonzai_j['label_bonzaiboost'], bert_j['label_bert'])
                    if bert_j['label_bert'] != 'O':
                        token['label'] = bert_j['label_bert']
                    elif crf_j['label'] != 'None':
                        token['label'] = crf_j['label']
                    else:
                        token['label'] = bonzai_j['label_bonzaiboost']
                    print(token['label'])

            # elif crf_j['label'] in CLASSES_CRF:
            #     token['label'] = crf_j['label']
            # elif bonzai_j['label_bonzaiboost'] in CLASSES_BONZAI:
            #     token['label'] = bonzai_j['label_bonzaiboost']
            # else:
            #     token['label'] = crf_j['label']

            res['annotations'] += [token]

        full_list += [res]

    print(conflicts, "conflicts resolved")
    print(votes_nocrf, 'votes against crf')
    print(votes_nobonzai, 'votes against bonzai')
    print(votes_nobert, 'votes against bert')
    print(conflicts_heuristic, "by heuristic")

    with open('preds_final.json', 'w') as out_f:
        out_f.write(json.dumps(full_list, ensure_ascii=False))
