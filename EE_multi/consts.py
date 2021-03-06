from get_arguments import *
hp = get_args()

NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"

# for BERT
CLS = '[CLS]'
SEP = '[SEP]'

SCHEMA = {"ACE": {
             "TRIGGERS": ['Business:Merge-Org',
                          'Business:Start-Org',
                          'Business:Declare-Bankruptcy',
                          'Business:End-Org',
                          'Justice:Pardon',
                          'Justice:Extradite',
                          'Justice:Execute',
                          'Justice:Fine',
                          'Justice:Trial-Hearing',
                          'Justice:Sentence',
                          'Justice:Appeal',
                          'Justice:Convict',
                          'Justice:Sue',
                          'Justice:Release-Parole',
                          'Justice:Arrest-Jail',
                          'Justice:Charge-Indict',
                          'Justice:Acquit',
                          'Conflict:Demonstrate',
                          'Conflict:Attack',
                          'Contact:Phone-Write',
                          'Contact:Meet',
                          'Personnel:Start-Position',
                          'Personnel:Elect',
                          'Personnel:End-Position',
                          'Personnel:Nominate',
                          'Transaction:Transfer-Ownership',
                          'Transaction:Transfer-Money',
                          'Life:Marry',
                          'Life:Divorce',
                          'Life:Be-Born',
                          'Life:Die',
                          'Life:Injure',
                          'Movement:Transport'],
             "ARGUMENTS": ['Place',
                           'Crime',
                           'Prosecutor',
                           'Sentence',
                           'Org',
                           'Seller',
                           'Entity',
                           'Agent',
                           'Recipient',
                           'Target',
                           'Defendant',
                           'Plaintiff',
                           'Origin',
                           'Artifact',
                           'Giver',
                           'Position',
                           'Instrument',
                           'Money',
                           'Destination',
                           'Buyer',
                           'Beneficiary',
                           'Attacker',
                           'Adjudicator',
                           'Person',
                           'Victim',
                           'Price',
                           'Vehicle',
                           'Time'],
             "ENTITIES": ['VEH:Water',
                          'GPE:Nation',
                          'ORG:Commercial',
                          'GPE:State-or-Province',
                          'Contact-Info:E-Mail',
                          'Crime',
                          'ORG:Non-Governmental',
                          'Contact-Info:URL',
                          'Sentence',
                          'ORG:Religious',
                          'VEH:Underspecified',
                          'WEA:Projectile',
                          'FAC:Building-Grounds',
                          'PER:Group',
                          'WEA:Exploding',
                          'WEA:Biological',
                          'Contact-Info:Phone-Number',
                          'WEA:Chemical',
                          'LOC:Land-Region-Natural',
                          'WEA:Nuclear',
                          'LOC:Region-General',
                          'PER:Individual',
                          'WEA:Sharp',
                          'ORG:Sports',
                          'ORG:Government',
                          'ORG:Media',
                          'LOC:Address',
                          'WEA:Shooting',
                          'LOC:Water-Body',
                          'LOC:Boundary',
                          'GPE:Population-Center',
                          'GPE:Special',
                          'LOC:Celestial',
                          'FAC:Subarea-Facility',
                          'PER:Indeterminate',
                          'VEH:Subarea-Vehicle',
                          'WEA:Blunt',
                          'VEH:Land',
                          'TIM:time',
                          'Numeric:Money',
                          'FAC:Airport',
                          'GPE:GPE-Cluster',
                          'ORG:Educational',
                          'Job-Title',
                          'GPE:County-or-District',
                          'ORG:Entertainment',
                          'Numeric:Percent',
                          'LOC:Region-International',
                          'WEA:Underspecified',
                          'VEH:Air',
                          'FAC:Path',
                          'ORG:Medical-Science',
                          'FAC:Plant',
                          'GPE:Continent'],
             "POSTAGS": ['VBZ',
                         'NNS',
                         'JJR',
                         'VB',
                         'RBR',
                         'WP',
                         'NNP',
                         'RP',
                         'RBS',
                         'VBP',
                         'IN',
                         'UH',
                         'JJS',
                         'NNPS',
                         'PRP$',
                         'MD',
                         'DT',
                         'WP$',
                         'POS',
                         'LS',
                         'CC',
                         'VBN',
                         'EX',
                         'NN',
                         'VBG',
                         'SYM',
                         'FW',
                         'TO',
                         'JJ',
                         'VBD',
                         'WRB',
                         'CD',
                         'PDT',
                         'WDT',
                         'PRP',
                         'RB',
                         ',',
                         '``',
                         "''",
                         ':',
                         '.',
                         '$',
                         '#',
                         '-LRB-',
                         '-RRB-']},
          "ERE": {
             "TRIGGERS": ['transaction:transfermoney',
                          'movement:transportperson',
                          'conflict:attack',
                          'personnel:endposition',
                          'contact:correspondence',
                          'movement:transportartifact',
                          'contact:meet',
                          'manufacture:artifact',
                          'life:die',
                          'conflict:demonstrate',
                          'contact:contact',
                          'transaction:transaction',
                          'life:injure',
                          'justice:arrestjail',
                          'transaction:transferownership',
                          'personnel:startposition',
                          'contact:broadcast',
                          'personnel:elect'],
             "ARGUMENTS": ['thing',
                           'recipient',
                           'person',
                           'giver',
                           'artifact',
                           'audience',
                           'entity',
                           'crime',
                           'destination',
                           'target',
                           'attacker',
                           'place',
                           'victim',
                           'agent',
                           'origin',
                           'time',
                           'money',
                           'position',
                           'beneficiary',
                           'instrument'],
             "ENTITIES": ['weapon',
                          'FAC:specificGroup',
                          'GPE:specificIndeterminate',
                          'LOC:specificIndeterminate',
                          'PER:specificGroup',
                          'age',
                          'vehicle',
                          'GPE:specificGroup',
                          'crime',
                          'ORG:nonspecific',
                          'url',
                          'title',
                          'LOC:specificIndividual',
                          'FAC:specificIndeterminate',
                          'PER:specificIndividual',
                          'LOC:nonspecific',
                          'ORG:specificIndeterminate',
                          'FAC:specificIndividual',
                          'PER:specificIndeterminate',
                          'ORG:specificGroup',
                          'time',
                          'FAC:nonspecific',
                          'money',
                          'ORG:specificIndividual',
                          'commodity',
                          'GPE:specificIndividual',
                          'LOC:specificGroup',
                          'PER:nonspecific',
                          'GPE:nonspecific'],
             "POSTAGS": ['LS',
                         'NNS',
                         'NNP',
                         'DER',
                         'DT',
                         'IJ',
                         'ADJ',
                         'RBR',
                         'FW',
                         'MSP',
                         '_',
                         'SCONJ',
                         'JJS',
                         'VBG',
                         '-LRB-',
                         'LC',
                         'VBZ',
                         'BA',
                         'VBP',
                         'VERB',
                         'AUX',
                         'PRP$',
                         'DEV',
                         ',',
                         'DET',
                         '-RRB-',
                         'PDT',
                         'VB',
                         'PRP',
                         'NR',
                         'VA',
                         'CS',
                         ':',
                         'OD',
                         'ADV',
                         'AD',
                         'WP$',
                         'WRB',
                         '``',
                         'MD',
                         'EX',
                         'CONJ',
                         '$',
                         'WP',
                         'NUM',
                         'PN',
                         '\'\'',
                         'RBS',
                         'LB',
                         'NOUN',
                         'VE',
                         'NNPS',
                         'URL',
                         'IN',
                         'NT',
                         '#',
                         'VV',
                         'JJR',
                         'ADP',
                         'RP',
                         'WDT',
                         'SYM',
                         'VC',
                         'ETC',
                         'PART',
                         'X',
                         'P',
                         'SB',
                         'VBD',
                         'CD',
                         'PU',
                         'AS',
                         'DEG',
                         'NN',
                         'CC',
                         'PRON',
                         '.',
                         'VBN',
                         'JJ',
                         'PUNCT',
                         'RB',
                         'INTJ',
                         'DEC',
                         'TO',
                         'SP',
                         'POS',
                         'PROPN',
                         'M',
                         'UH']},
          "BETTER": {
              "TRIGGERS": {
                  "use_quad": ['helpful_material',
                                'helpful_verbal',
                                'harmful_material',
                                'harmful_verbal',
                                'neutral_material',
                                'neutral_verbal',
                                'helpful_neutral',
                                'harmful_neutral',
                                'neutral_neutral',
                                'neutral_both',
                                'both_neutral'
                                'both_material',
                                'both_verbal',
                                'helpful_both',
                                'harmful_both',
                                'both_both',
                                'neutral_unk',
                                'unk_neutral',
                                'both_unk',
                                'unk_unk',
                                'unk_both',
                                'unk_material',
                                'unk_verbal',
                                'helpful_unk',
                                'harmful_unk',
                                'SPECIFIED_NOT'],
                  "no_quad": ["unk_unk"]},
              "ARGUMENTS": ['agent',
                            'patient'],
              "ENTITIES":  ['QUANTITY',
                            'EVENT',
                            'ORDINAL',
                            'FAC',
                            'TIME',
                            'ORG',
                            'GPE',
                            'PERCENT',
                            'LANGUAGE',
                            'CARDINAL',
                            'VEH',
                            'None',
                            'NORP',
                            'PRODUCT',
                            'WORK_OF_ART',
                            'MONEY',
                            'PERSON',
                            'LOC',
                            'PER',
                            'DATE',
                            'WEA',
                            'LAW'],
              "POSTAGS": ['NFP',
                          '$',
                          ',',
                          'RBR',
                          'WRB',
                          'AFX',
                          'RBS',
                          'VBG',
                          'WP$',
                          '.',
                          'PRP$',
                          'JJ',
                          'HYPH',
                          '``',
                          'NN',
                          'FW',
                          'VBZ',
                          'MD',
                          'UH',
                          ':',
                          'WDT',
                          'POS',
                          'DT',
                          'RB',
                          'EX',
                          'PRP',
                          'NNPS',
                          'CC',
                          'VBN',
                          'XX',
                          'VBP',
                          "''",
                          'WP',
                          'NNS',
                          'JJR',
                          'NNP',
                          'RP',
                          'SYM',
                          'IN',
                          'VB',
                          'JJS',
                          'TO',
                          '-RRB-',
                          'CD',
                          'VBD',
                          'PDT',
                          '-LRB-',
                          'NUM',
                          'NOUN',
                          'ADP',
                          'CCONJ',
                          'PUNCT',
                          'DET',
                          'PRON',
                          'VERB',
                          'X',
                          'ADJ',
                          'AUX',
                          'PART']
              }
          }


if hp.schema_type == "BETTER":
    if hp.use_quad:
        TRIGGERS = SCHEMA[hp.schema_type]["TRIGGERS"]["use_quad"]
    else:
        TRIGGERS = SCHEMA[hp.schema_type]["TRIGGERS"]["no_quad"]
else:
    TRIGGERS = SCHEMA[hp.schema_type]["TRIGGERS"]


ARGUMENTS = SCHEMA[hp.schema_type]["ARGUMENTS"]
ENTITIES = SCHEMA[hp.schema_type]["ENTITIES"]
POSTAGS = SCHEMA[hp.schema_type]["POSTAGS"]
