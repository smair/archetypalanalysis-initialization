# file name format is result_DATA_K_INIT_M_PREPROCESSING_REPS_MAXITER_M.npz
M = 1000.0

STYLE = {'Uniform':{'color':'tab:red'},
         'FurthestFirst':{'color':'tab:purple'},
         'FurthestSum':{'color':'tab:blue'},
         'KMpp':{'color':'tab:orange'},
         'KMppMC':{'color':'tab:orange'},
         'AApp':{'color':'tab:green'},
         'AAppMC':{'color':'tab:green'},
         'AAcoreset':{'color':'tab:brown'}}
STYLE_M = ['--','-.',':',':']

# data set size if we want to normalize
n_for = {
         'BankProblem1':4971,
         'BankProblem2':12456,
         'BankProblem3':19939,
         'airfoil':1503,
         'california':20640,
         'concrete':1030,
         'covertype':581012,
         'ijcnn1':49990,
         'kdd-protein':145751,
         'miniboone':130064,
         'pose':35832,
         'rna':488565,
         'song':515345,
         'sun-attribute':14340,
         'fma':106574}

K_for = {'BankProblem1':[15,25,50,75,100],
         'BankProblem2':[15,25,50,75,100],
         'BankProblem3':[15,25,50,75,100],
         'airfoil':[15,25,50,75,100],
         'california':[15,25,50,75,100],
         'concrete':[15,25,50,75,100],
         'covertype':[15,25,50,75,100],
         'ijcnn1':[15,25,50,75,100],
         'kdd-protein':[15,25,50,75,100],
         'miniboone':[15,25,50,75,100],
         'pose':[15,25,50,75,100],
         'rna':[15,25,50,75,100],
         'song':[15,25,50,75,100],
         'sun-attribute':[15,25,50,75,100],
         'fma':[15,25,50,75,100]}

repetitions_for = {'BankProblem1':30,
                   'BankProblem2':30,
                   'BankProblem3':30,
                   'airfoil':30,
                   'california':30,
                   'concrete':30,
                   'covertype':15,
                   'ijcnn1':30,
                   'kdd-protein':30,
                   'miniboone':30,
                   'pose':30,
                   'rna':30,
                   'song':15,
                   'sun-attribute':30,
                   'fma':15}

time_normalizer_for = {'BankProblem1':1.0,
                       'BankProblem2':1.0,
                       'BankProblem3':1.0,
                       'airfoil':1.0,
                       'california':1.0,
                       'concrete':1.0,    # normalize time by 1 to keep seconds
                       'covertype':60.0,  # normalize time by 60 to get minutes
                       'ijcnn1':1.0,
                       'kdd-protein':60.0,
                       'miniboone':60.0,
                       'pose':1.0,
                       'rna':60.0,
                       'song':60.0,
                       'sun-attribute':1.0,
                       'fma':60.0}

LABEL_RENAME = {'Uniform':'Uniform',
                'FurthestFirst':'FurthestFirst',
                'FurthestSum':'FurthestSum',
                'KMpp':r'$k$-Means++',
                'AApp':'AA++',
                'AAppMC1':'AA++MC 1\%',
                'AAppMC5':'AA++MC 5\%',
                'AAppMC10':'AA++MC 10\%',
                'AAppMC20':'AA++MC 20\%',
                'AAcoreset':'AAcoreset'}

DATASET_RENAME = {'airfoil':'Airfoil',
                  'BankProblem1':'BankProblem1',
                  'BankProblem2':'BankProblem2',
                  'BankProblem3':'BankProblem3',
                  'california':'California',
                  'concrete':'Concrete',
                  'covertype':'Covertype',
                  'ijcnn1':'Ijcnn1',
                  'kdd-protein':'KDD-Protein',
                  'miniboone':'MiniBooNE',
                  'pose':'Pose',
                  'rna':'RNA',
                  'song':'Song',
                  'sun-attribute':'SUN Attribute',
                  'fma':'FMA'}
