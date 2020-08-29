import numpy as np
import requests


def get_assay_results(aid, tids=None):
    assay_results = []
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}'
    request = requests.get(f'{url}/sids/json')
    sids = request.json()['InformationList']['Information'][0]['SID']
    limit = 10000
    batches = [sids[i * limit:(i + 1) * limit] for i in range((len(sids) + limit - 1) // limit)]
    for batch in batches:
        request = requests.post(f'{url}/json', data={'sid': ','.join(map(str, batch))})
        data = request.json()['PC_AssaySubmit']['data']
        for compound in data:
            if tids is None:
                props = [list(prop['value'].values())[0] for prop in compound['data']]
            else:
                props = [list(prop['value'].values())[0] for prop in compound['data'] if prop['tid'] in tids]
            assay_results.append([compound['sid']] + props)

    return np.array(assay_results, dtype=object)


def get_smile(sids):
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/substance/sid/cids/json'
    request = requests.post(url, data={'sid': ','.join(map(str, sids))})
    compounds = request.json()['InformationList']['Information']
    smile_table = np.ndarray((len(compounds), 2), dtype=object)
    for index in range(smile_table.shape[0]):
        smile_table[index, 0] = compounds[index]['CID'][0] if 'CID' in compounds[index] else None

    cids = smile_table[smile_table[:, 0] != None, 0].astype(int)
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/CanonicalSMILES/json'
    request = requests.post(url, data={'cid': ','.join(map(str, cids))})
    smiles = request.json()['PropertyTable']['Properties']
    smiles_index = 0
    for index in range(smile_table.shape[0]):
        smile = smiles[smiles_index]
        if smile_table[index, 0] == smile['CID']:
            smile_table[index, 1] = smile['CanonicalSMILES']
            smiles_index += 1
        else:
            smile_table[index, 1] = None

    return smile_table
