from comet_ml.api import API
import numpy as np
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.style.use('seaborn')
#rc('text', usetex=True)
rc('font', family='serif')
import seaborn as sns
sns.set_palette('Paired')

# comet
api_key = 'tHDbEydFQGW7F1MWmIKlEvrly'
workspace = 'aguerra'
project_name = 'arsac-test'
comet_api = API(api_key=api_key)

# savgol filter
SMOOTH = True
WINDOW = 71
POLY_DEG = 3

REWARD_KEY = 'train_Avg. Episode_Reward'
LOG_SCALE_KEY = 'train_AR log scale'

def get_returns(experiment, metric):
    """
    Obtains the (eval) returns from an experiment.

    Args:
        experiment (Experiment): a comet experiment object

    Returns a numpy array [n_episodes,] of cumulative rewards (returns).
    """
    # asset_list = experiment.get_asset_list()
    # returns_asset_list = [a for a in asset_list if '_reward' in a['fileName']]
    # if len(returns_asset_list) > 0:
    #     returns = []
    #     steps = []
    #     for asset in returns_asset_list:
    #         data = experiment.get_asset(asset['assetId'], return_type='json')
    #         returns.append(sum(data))
    #         steps.append(float(asset['step']))
    #     steps = np.array(steps)
    #     print(steps[-5:])
    # else:
    returns_asset_list = experiment.get_metrics(metric)
    returns = [float(a['metricValue']) for a in returns_asset_list]
    steps = np.array([float(a['step']) for a in returns_asset_list])
    # # Jenky fix for old arsac tests
    # if len(returns) > 100:
    #     returns = returns[1::2]
    #     steps = np.array(range(len(returns))) * 1e3
    return np.array(returns), steps / 1e3


def aggregate_returns(returns_list):
    """
    Aggregates (mean and std.) the returns from multiple experiments.

    Args:
        returns_list (list): a list of numpy arrays containing returns

    Returns numpy arrays of mean and std. dev of returns.
    """
    if len(returns_list) == 1:
        return returns_list[0], None
    min_episodes = min([exp.shape[0] for exp in returns_list])
    returns_array = np.stack([r[:min_episodes] for r in returns_list])
    returns_mean = returns_array.mean(axis=0)
    returns_std = returns_array.std(axis=0)
    return returns_mean, returns_std


def plot_mean_std(mean, std, label):
    if SMOOTH:
        if len(mean) > WINDOW:
            mean = savgol_filter(mean, WINDOW, POLY_DEG)
        if std is not None and len(std) > WINDOW:
            std = savgol_filter(std, WINDOW, POLY_DEG)
    plt.plot(mean, label=label)
    if std is not None:
        plt.fill_between(np.arange(mean.shape[0]), mean - std, mean + std, alpha=0.2)


def plot_rewards(env_exp_dict, save_folder, metric):
    if metric == REWARD_KEY:
        plt.ylim(ymax=1010, ymin=-10)
    for inference_type in env_exp_dict:
        returns_list = []
        for exp_key in env_exp_dict[inference_type]:
            if exp_key == '':
                continue
            experiment = comet_api.get_experiment(project_name=project_name,
                                              workspace=workspace,
                                              experiment=exp_key)
            if experiment is not None:
                returns, steps = get_returns(experiment, metric)
                returns_list.append(returns)
            else:
                print("Invalid Comet ML Experiment Key provided: " + exp_key)

        if len(returns_list) > 1:
            mean, std = aggregate_returns(returns_list)
            if inference_type == "SAC (2x256)":
                end_mean_array = [mean[-1]] * mean.shape[0]
                plt.plot(end_mean_array, '--', color='black')
            else:
                plot_mean_std(mean, std, inference_type)
        elif len(returns_list) == 1:
            print(len(returns_list))
            plt.plot(savgol_filter(returns_list[0], WINDOW, POLY_DEG), label=inference_type)
    plt.legend(fontsize=12)
    #plt.xlabel(r'Steps $\times 1,000$', fontsize=15)

    plt.xlabel('Eval Episode', fontsize=15)
    if metric == REWARD_KEY:
        plt.ylabel('Cumulative Reward', fontsize=15)
    else:
        plt.ylabel('Avg. Log(scale)')
    plt.title(env, fontsize=20)
    name = env.replace("w/", "_").replace(" ", "")
    end_name = "_rewards" if metric == REWARD_KEY else "_log_scale"
    plt.savefig(save_folder + "pdfs/" + name + end_name + '.pdf')
    plt.savefig(save_folder + "jpgs/" + name + end_name + '.jpg')
    plt.clf()

# DM Control Transfer tasks
walker_transfer_dict = {"SAC": ['21ed3d77773e4249b171a321a2bdcd07'],
                 "SAC w/ base transfer": ['2ce5802c9ea845fc9156514c21b53c02'], # Transferred from 0c00a35dea5e425d8af55d3804b3c7b0
                 "ARSAC": ['0e5f9791511242eaa54f175af46a42f7'],
                 "ARSAC w/ flow transfer": ['cf8a735331fc4fc689943cf1581f3c79'], # Transferred from 157216c90f8e400bab5264211ede1646
                 "ARSAC w/ base + flow transfer": ['a131e09fef224a27ab9d5050aa7d2dd5']} # Transferred from 157216c90f8e400bab5264211ede1646
quadruped_transfer_dict = {"SAC": ['b677035a0e044c5db3a9f598ccc3de90',
                                    '3e0e68b5d2fc4ecdb397a8b2e012bf44',
                                    '1f0f42bbebd8418b81e01c8f21e09d73'],
                 "SAC w/ base transfer": ['99647116771d457bb752d0f79d30b63f',
                                          '91dd4c2b45b54a9db2c5e155f58b5ece',
                                          '2ecddfc24e784aa6b67542403dc2078a'], # Transferred from 829f7aff24e041ef94829e266147965f
                 "ARSAC-5": ['d2e43dbd1edc4f2a8074d60f92f8191a',
                             'a7c6092e814846eb94d0393f896b83b0',
                             '0aa732f33af84377b61f25a1c9971575'],
                 "ARSAC w/ flow transfer": ['3a5ef39381d54d31a8d689bb4171beda',
                                            '9b980baa1e9d4d89af3b407e566947cf',
                                            '8538afd5185840e3a93d55c408b56f13'], # Transferred from 4e96ec38641a4ed59e31050e69811888
                 "ARSAC w/ base + flow transfer": ['2c1de67cbc9e47ee821e06a336b032f0',
                                                   'be7c1764a0c64818ad166461ac46e9aa',
                                                   'ef1c058aa07f424c8c2366b967c53e93']} # Transferred from 4e96ec38641a4ed59e31050e69811888

# DM Transfer Control Tasks, this time with Auto-Entropy Tuning and Hidden Size 256
walker_transfer_dict2 = {"SAC": ['707acf6bfd714c6582e737db0191743f',
                                  '1cd7e219a5fb46409eab5ca2bd476420',
                                  '76eb27add9a04d4c8867379ea0d20984',
                                  'dd362b0740db449b8a156c9edf4e2b49',
                                  'ba7f1b039bc84c99b228924d6f2e4bc2'],
                 "SAC w/ base transfer": ['333e459c697c454b9a63fd0f4f14e201',
                                          'c9f13043b28446febee921d2eb31406b',
                                          '6847c05ab65e4cf38b5e854015e821b8',
                                          '2cf06ade668146f39263229dcca0c033',
                                          '9b686f67af08417d83c1d7697981a815'], # Transferred from 0f1c2f68e5124ff282283fd6a28961f6
                 "ARSAC-5": ['cd2b9f003e404a8daff56dea22e0bcb3',
                             'a70d6e85bebb48afa5ab7cd90a462f84',
                             'a6bffea54b32423da7f4a2d2ffc6b0b8',
                             'b549680b06e74dd884bb5dc352e8ed97',
                             '73cdbd54d8914d3bbefd4597e0eb89d5'],
                 "ARSAC w/ flow transfer": ['db603b5dabae46eea227c2232d4ec3a5',
                                            '5ff7fd4a1cdb4e57a33e6349f5e288ef',
                                            '1ffb2517534f4387982cfa4aae96baa1',
                                            'ff7ef746ccff4d3da3be6aced21ca938',
                                            '9a46b4a350b54610afcea3b04cd9fd1f'], # Transferred from 113982d9ab8b4faf849ef5a2efb712f5
                 "ARSAC w/ base + flow transfer": ['5d271e81a6434a0a902155dbef1c71df',
                                                   '3c28e9c888b943d78c5d964ed15fc254',
                                                   '7bb3056fa6724bb29e56e0de49df9a6d',
                                                   'cae9da2c19c5423a9afc5d02866a97ae',
                                                   'ad725ff21059433b9e5cad6db3b193d6']} # Transferred from 113982d9ab8b4faf849ef5a2efb712f5
hopper_transfer_dict2 = {"SAC": ['3945a372be2645c7a6b8efe2951cc3d1',
                                 '62109befb53c4f658a7f21a28017c7e1',
                                 '3e43964bcab54d5fb88f815f29cfb305',
                                 '8c85abb6e4d84841b9f7fcc52bc9b2cb',
                                 '528ec1181aa643a6a454e7eae55a435f'],
                 "SAC w/ base transfer": ['6d872554e6924e77983d88376b8c633d',
                                          'd42be0cc391c4285a54c9566d26d4a9f',
                                          '7377786b1c4f4a9a8eb604418b0d9eb4',
                                          '81ad6cc5826540c3bcddbb892070f6bd',
                                          'e6c908788d394051bd2cb77c81dadfc3'], # Transferred from d3649569dbf64475a8e1816261a773c1
                 "ARSAC-5": ['f34a21792fd049af98794b666cf292ff',
                                     '789d8441dc6248e280a7026326754832',
                                     '137012c98cb7417c983c93d214ef1fbf',
                                     'e0300ac61e1345a8a3e13cb5c0876035',
                                     '50e034bba4894b739489478fe3a75ed5'],
                 "ARSAC w/ flow transfer": ['35ba36ee111b41929baf3a2591afce49',
                                            '598bd0127114433abe950a6eb3b08ebb',
                                            '7cba40d750c44d96bc0687e07e57f271',
                                            'f5a6a9ca7f534d7ba95ff7d61b7bc666',
                                            '7bc4b5be54a64416b5bedaea4ae820c6'], # Transferred from e8218e27d9c74d659bc62b9242ab2de2
                 "ARSAC w/ base + flow transfer": ['0e9eb52affc14456bc2cba817cf4796f',
                                                   '7fda4553905143b0a53326b2209d876d',
                                                   'f5486cde8149475cb9d161c3e275016f',
                                                   '4501031cc6c749cdaae21c408c6f3a08',
                                                   'f601f16b573e4a889c29402b207fd7b8']} # Transferred from e8218e27d9c74d659bc62b9242ab2de2

quadruped_transfer_dict2 = {"SAC": ['a3116a2d342049258b3b4c078b0900a9',
                                    '8c36041ad20144c0bed4372e5bee7830',
                                    '638e1f47b4554c3abcb99e8bd7eb1c7c',
                                    'e9f62562f0f3404e8e6adc0a87795c41',
                                    '8679973fbc8a4147980d1e208ffc99f0'],
                 "SAC w/ base transfer": ['d13532a1f404496a8a33597e1f0105df',
                                          '260bdd5da63f485abf67f0efd20e677e', #<- this experiment may have incomplete data
                                          'e3a43f3a0be24a61ba21c646b2966f93', #<- this experiment may have incomplete data
                                          '9fdb83871cd64a52861c1e87909c130d', #<- this experiment may have incomplete data
                                          'f000eb14597e4ed1957981c76ef897d7'], # Transferred from 63f05e87479d48869c2f0aac465700ec
                 "ARSAC-5": ['739af68727ff409eaf20e7cca5e9c233',
                              'a5af0b02e7a247b391f6069cfeab9dbb',
                              '025da2ce2e7e4768a36cba0011ea05f0',
                              '3e9c9ecf75d94e2fbea7239e534395b9',
                              'be93ba6eb8414f2f9d6969e7f8dd960b'],
                 "ARSAC w/ flow transfer": ['9574452c9b474fe99eb7efd5170e0c79',
                                            '8c8800797c474819a7cec4265f63afdc', #<- this experiment may have incomplete data
                                            '43037f0e4b47428cb1cf2be773ed7766',
                                            '54fc235e8fbf4302b0d9065f020ef95c',
                                            '2f97bea10b1d455e9b7179d88171488c'], # Transferred from a8f7092ff8de48cc97b7e8dd2e18849a
                 "ARSAC w/ base + flow transfer": ['33c6dc2edceb49a5b67316c981a4dec7',
                                                   'e71f43683a404a16a6192e60c8ec2f43',
                                                   '65bd6596443a47e5acb6bc088f4a40b5',
                                                   '6e290fcc6e2948d2980aabd4ffe28ed0', #<- this experiment may have incomplete data
                                                   'ae1f7e8c878c4dcbb82e5850aef9179a']} # Transferred from a8f7092ff8de48cc97b7e8dd2e18849a

# DM Control Transfer tests, this time with hidden dim 32, AutoEnt on, 256 Batch Size
walker_transfer_dict3 = {"SAC" : ["3193d027636e4593a8ba71a84ee21638",
                                    "6528837f8fc94920a472fe827f98ca63",
                                    "8e8450864b264fe18ed309a9e6fc6866",
                                    "b6e6f26e60794ce69943c75138c7731c",
                                    "29af5554dec04009a3e56a581dd420a8"],
                 "SAC w/ base transfer": ['301744b1fde8410895c22c614f0ecf61',
                                          'ac321ae9df2e4f40a5d000ea0ef274d4',
                                          '7d5e09f9d689494eb01651d6e41a58a1',
                                          '32ce54fcfaf141dc8bf170a677ed34f8',
                                          'ad4dee742c5744cd82c4107d6b85d1bf'],
                 "ARSAC": ["74e14df64e214d8ab68fcdb91bd44cb8",
                            "1cf5297adce049699f860e1438707bab",
                            "5d1f3bea197a4530b8cf22ceb8d44f0f",
                            "deb002867c1747b39e62bbbf182b8e68",
                            "ed83960ca38e4807be3a659caa9302e5"],
                 "ARSAC w/ flow transfer": ['2cd4a0b7d05c4fe38d3064e8d11ad66c',
                                            '7795ca6c124e424b89b6b03480a9048f',
                                            '0eab530d9dc144889dda032648ea1930',
                                            '0ce65345a8e74f5dae2731cc1d05bdaa',
                                            'b507afbc76104a388f6dd3dc759d2a94'],
                 "ARSAC w/ base + flow transfer": ['7a1a6e5a81c94b6bbb23ed84e525904e',
                                                   '3097ffdbbdc4439a8ee9b420ecd50ab2',
                                                   'a35e002acd4a4bf6aa162cdc594b8520',
                                                   '1cd85c5aa2114ecd8c6380e1d7e198be',
                                                   '0c4129b85af347fda4202feb7ec796aa']}

quadruped_transfer_dict3 = {"SAC" : ["0eeb188dba754404a41ed786a52d5ac8",
                                    "55bb145d0b4b4921a60a154504160e0b",
                                    "cc69bfb3e07d433dbe49977dc4a34b79",
                                    "2e1b579760ac4ff68adf3a4e11181803",
                                    "4f295c37b35a4269a016390cd522ffe3"],
                 "SAC w/ base transfer": ['2daf01fa614745fab88ade4b0b2413a4',
                                          'f2ed356fefc54f708673872ca7f451d5',
                                          '12e67f788a4147f6a5eaecdb27bb0740',
                                          '34fa07b91d24403b8b939a85d822f0c2',
                                          '737aac92b47a4a669ee053dda10e751b'],
                 "ARSAC": ["56911a04f19347898de823f159cf0f52",
                            "41b6b5e0401945cfb26f6bbe201720a3",
                            "bb9705e69aad45cd85395eefb1f267b0",
                            "a56a0c85760c466bb24daa092041a4fe",
                            "674eca50665948a089295335e8de13b0"],
                 "ARSAC w/ flow transfer": ['7b5892c88a91488fbb55d92c9d518b52',
                                            'f648f350218b41db9a807b70823b07e3',
                                            '58ff405864be4a2fb3c95e84347b73c8',
                                            'ea39b604d2e347a491bbf208820179a5',
                                            'e7e728492a454b4a9494f70a3cb84633'],
                 "ARSAC w/ base + flow transfer": ['daf94c0c85414a38a89c10b461bb3109',
                                                   'd3bcf58af13746b2b345d44977c9cbc9',
                                                   'e909b4b55e3541a1a435502bd40f06af',
                                                   '2779549c387f4fc688b10fbfaf7edd30',
                                                   '35c9d04f91ed47ca8b6d1d0b78245cb9']}

# DM Control Base Tests (Batch size 128, ARSAC-3, hidden dim 32)
walker_walk_base_dict = {"SAC": ['5486ada760c640f7b5fbdd3680ce8258'],
                         "ARSAC-3": ['848a8336f98343dca7e09ff80e221dba']}
walker_run_base_dict = {"SAC": ['a2277a87765e4ddd8d9af641a8ce97cf'],
                         "ARSAC-3": ['f2e735fbf6d0490b8a778371d2af69f3']}
quadruped_walk_base_dict = {"SAC": ['502d3394b8f04710aeeda0522d2765e9'],
                         "ARSAC-3": ['22ece5090c424bf7ba1fd0a00ec57c78']}
quadruped_run_base_dict = {"SAC": ['8c3a67964315417d964ccc86ff2b77f7'],
                         "ARSAC-3": ['88556932cf1341d4bbd76f0bda65a71e']}
hopper_stand_base_dict = {"SAC": ['27553ab85f5d4f2981339ecb244ed92a'],
                         "ARSAC-3": ['6fd32d97a7214344b16da9b6b6ae6d71']}
hopper_hop_base_dict = {"SAC": ['e4c400fa2c654d739952138653271a55'],
                         "ARSAC-3": ['bcdbbdd174764aba84742d2f9122f789']}

# DM Control Base Tests (Batch Size 256, ARSAC-5, hidden dim 32)
walker_walk_base_dict2 = {"SAC": ['b1acc1d1812b490c91437d8735f37911', # 1
                                  '25cabb2bf78b4a26847cdbf07fee1f73', # 2
                                  '8671749406ad4e4a97372d36e28d1bd0'], # 3
                         "ARSAC-5": ['3b7564c7ca044faeaf459e7aa5635b98',
                                     '14007a1391ca4758a7c07250957902c6',
                                     'c78c27f6b2854a38a9100a688c553b39']}
walker_run_base_dict2 = {"SAC": ['3f271263d96b45009a19b32aa816260b',
                                 'ac28eb5a38b3468b9de3c641ca3f86c2',
                                 '5af66ac575284976a624b25371d99ef6'],
                         "ARSAC-5": ['91cb40a43bb64e23854feb8efe76db8c',
                                     '1441bfaa31954a239f0d3fa0cb789b1a',
                                     '437f1e393c6e4ed5aced57583bda3460']}
quadruped_walk_base_dict2 = {"SAC": ['ea66866a0f4440c4b5c9a979df987ae7',
                                     'a7d9fc48ff7a49fc92faa6d83980ece1',
                                     '475848cab7d24c4cab0bde5c7f4f24d3'],
                         "ARSAC-5": ['68fd82b266614f3db1430cd7c63eaf7e',
                                     '571019e4d63a46b6bab1069b150d3fa9',
                                     '12b06528d71243bbb92516121cdbdf80']}
quadruped_run_base_dict2 = {"SAC": ['af54967243e748cf94d62341712c8336',
                                    '7d869b663a584665b2e7958342a44711',
                                    'c32b0ffc37724d178efa886a1b94cfa6'],
                         "ARSAC-5": ['6f9ba61406c54cefa9cdde52b2766e90',
                                     '8c7010391eea4e0e94a0bcc900b1e550',
                                     '6fe0d6da6c7247d0ae1e2c73959323c6']}
hopper_stand_base_dict2 = {"SAC": ['c2eecc92b7034fab97c712a8d372e9a5',
                                   '6cbfa580e0e349608eaead9286947edb',
                                   'ef9f5a9f20e640d289c21a32c66b6dae'],
                         "ARSAC-5": ['38a6cc65e68f4b9f93261d1a7fa5c13c',
                                     '5ad134f56f92420cb0ea7c308e7cb324',
                                     '9a082948d3cd4325808d1d48e9890c45']}
hopper_hop_base_dict2 = {"SAC": ['84fe85ab190e46829a5ea83ce7f71c66',
                                 '880a4442ab18430bb41d277e1da1d6a5',
                                 '2c09f071cd51428c94af79294e80da1f'],
                         "ARSAC-5": ['94b386e376124a1d8d157ce0a85c758a',
                                     '497aaa6f7cc1481d89339c2a1df0bfd7',
                                     '74404513a2a245c3adcff5b96b1c4254']}

# DM Control Base Tests (Batch Size 256, ARSAC-5, hidden dim 2x256)
# We only do quadruped and hopper since walker was able to learn with 1x32 hidden size
quadruped_walk_base_dict3 = {"SAC": ['829f7aff24e041ef94829e266147965f',
                                     '511fb31a0a6b48fcae11a43c436616f2',
                                     'a73054aa200b42d08978a8487937642d'],
                         "ARSAC-5": ['4d75ad875f5140728c7be89bb0bb96e4',
                                     'cc937d7005274803a640a68c4aa8367c',
                                     'a966793edf034ac79fb63918d410f450']}
quadruped_run_base_dict3 = {"SAC": ['b677035a0e044c5db3a9f598ccc3de90',
                                    '3e0e68b5d2fc4ecdb397a8b2e012bf44',
                                    '1f0f42bbebd8418b81e01c8f21e09d73'],
                         "ARSAC-5": ['d2e43dbd1edc4f2a8074d60f92f8191a',
                                     'a7c6092e814846eb94d0393f896b83b0',
                                     '0aa732f33af84377b61f25a1c9971575']}
hopper_stand_base_dict3 = {"SAC": ['13f01b2093de4e058b5ab71594834d2b',
                                   '9a23165ed514410ca6e8a81f991cd037',
                                   'e6552a8917514a4a888f4d1c3dbe0621'],
                         "ARSAC-5": ['b5cecb25688340f48f691b9cbcf73a6b',
                                     'd75d30f5d83d423ea196c68fa5a93a66',
                                     '5a737d5d786d4c659c76978c3f860ac4']}
hopper_hop_base_dict3 = {"SAC": ['ea594f54e21b4c01921cc2dbfd3f7189',
                                 '09e56e24a85c446fbf76a6c05facceae',
                                 '8a82bace2dbf443fa36051ae08cccd05'],
                         "ARSAC-5": ['b70145be1cdb45a1bdc97b1a849f0d93',
                                     'e73dcb0f4d62404cba985344b31a6a26',
                                     '908fcee3ac8d4156a2e83f8639df8111']}
# Same tests, but with auto-entropy tuning. Done on more environments
walker_walk_base_dict4 = {"SAC": ['2ea19479726d44dfa8cd06d94f620178',
                                  '0f1c2f68e5124ff282283fd6a28961f6',
                                  '155446f560724c07837e42841527d9d0',
                                  '1b4f31622c884af2adac83fc4df5b6f6',
                                  'f11b9050bd1c4ffe93cd049d05dd7820'],
                          "ARSAC": ['113982d9ab8b4faf849ef5a2efb712f5',
                                    '0db2b5f019f3424eb0c17a34265932e3',
                                    '72775f63da174d9889c956d34f8335b1',
                                    '279206ac947343439547f827a55ca309',
                                    '47637c6e9ca646dd8fe6c4eb974ab7c9']}
walker_run_base_dict4 = {"SAC": ['707acf6bfd714c6582e737db0191743f',
                                  '1cd7e219a5fb46409eab5ca2bd476420',
                                  '76eb27add9a04d4c8867379ea0d20984',
                                  'dd362b0740db449b8a156c9edf4e2b49',
                                  'ba7f1b039bc84c99b228924d6f2e4bc2'],
                          "ARSAC": ['cd2b9f003e404a8daff56dea22e0bcb3',
                                    'a70d6e85bebb48afa5ab7cd90a462f84',
                                    'a6bffea54b32423da7f4a2d2ffc6b0b8',
                                    'b549680b06e74dd884bb5dc352e8ed97',
                                    '73cdbd54d8914d3bbefd4597e0eb89d5']}
quadruped_walk_base_dict4 = {"SAC": ['63f05e87479d48869c2f0aac465700ec',
                                     '38e240215f2c408aad8e29f9e2889a44',
                                     '7a51337b3690473283028d027d047aa7',
                                     'bde9ef35ca214c27abeb7c5ee332d82d',
                                     '1fd9cb29169340dd9f8295fe86cea399'],
                         "ARSAC-5": ['a8f7092ff8de48cc97b7e8dd2e18849a',
                                     '4e96ec38641a4ed59e31050e69811888', # Used for Transfer experiments as well
                                     '04a9f3dacea34812aba6fcfaa24efe70',
                                     '024073a18174487195b1776bad8fe75f',
                                     'e56565f75b0b4ee291d10a44362df8cd']}
quadruped_run_base_dict4 = {"SAC": ['a3116a2d342049258b3b4c078b0900a9',
                                    '8c36041ad20144c0bed4372e5bee7830',
                                    '638e1f47b4554c3abcb99e8bd7eb1c7c',
                                    'e9f62562f0f3404e8e6adc0a87795c41',
                                    '8679973fbc8a4147980d1e208ffc99f0'],
                            "ARSAC": ['739af68727ff409eaf20e7cca5e9c233',
                                      'a5af0b02e7a247b391f6069cfeab9dbb',
                                      '025da2ce2e7e4768a36cba0011ea05f0',
                                      '3e9c9ecf75d94e2fbea7239e534395b9',
                                      'be93ba6eb8414f2f9d6969e7f8dd960b']}
hopper_stand_base_dict4 = {"SAC": ['d3649569dbf64475a8e1816261a773c1',
                                   'f2ab7089ecbd4626bdcc92682d7fbaa2',
                                   '64b30262c75046459f1e24c5ee4d1f3d',
                                   '285f149e035749e3a976e8424ff561f4',
                                   '390dec8c1ac4475798fa17c40f09e72d'],
                         "ARSAC-5": ['e8218e27d9c74d659bc62b9242ab2de2',
                                     'f06a4267d1e9456ea04aa257b71283c0',
                                     '84b578ed4c93400e9fa0ffce9239e98e',
                                     'ad9418fa5615497fb1917a072d7a7e23',
                                     '436a2e208a4f4ffa8827eff3234d61c4']}
hopper_hop_base_dict4 = {"SAC": ['3945a372be2645c7a6b8efe2951cc3d1',
                                 '62109befb53c4f658a7f21a28017c7e1',
                                 '3e43964bcab54d5fb88f815f29cfb305',
                                 '8c85abb6e4d84841b9f7fcc52bc9b2cb',
                                 '528ec1181aa643a6a454e7eae55a435f'],
                         "ARSAC-5": ['f34a21792fd049af98794b666cf292ff',
                                     '789d8441dc6248e280a7026326754832',
                                     '137012c98cb7417c983c93d214ef1fbf',
                                     'e0300ac61e1345a8a3e13cb5c0876035',
                                     '50e034bba4894b739489478fe3a75ed5']}
cheetah_run_base_dict4 = {"SAC": ['d4f8852ef3724b1b922a470f1faecc8e',
                                  '472c1e81b25c4578a5ccadc45f93255f',
                                  'fb194bd6fdd54d06bfa588118cd1451f',
                                  '9168d02d34c34b13bef04eecfbcc2bcf',
                                  '77350e72c5a047b5a45b18e77da2653a'],
                           "ARSAC": ['d249049426d04c44bbeb1a817e20369c',
                                     '978bfd18862d4bddb55748312fe82d5c',
                                     '88735d788b2f43c397bc90cc9d141a9e',
                                     '6440820fe6c44b93b17cf2ea3cfb0b44',
                                     '45c8f087f682459593b8286f366effb0']}
swimmer_swimmer6_base_dict4 = {"SAC": ['9261019f9dae4b2e905083be0f58ef7c',
                                        'f3577a6c31a147dfbfec6f85b93c7e76',
                                        '13ea38a54d2a45da8e260b8dfbfd05b7',
                                        '1ac80220920d4ed0930c00cc9a2500eb',
                                        '28a6559428ab49a381e5e9e11759c878'],
                               "ARSAC": ['6cd4bf639905460e8f23e3e9f42bd963',
                                         '96224b8d827c47dbb132dd8591223a4c',
                                         '74533d179112481995b49f1cbbfcad87',
                                         '61d70de25e674332aedbcc30308aedd7',
                                         '93f54dab1c584633bc4d7a530dcc357e']}

humanoid_stand_base_dict4 = {"SAC": ['bb483d97b2964d0b99935071cf3667e7',
                                     '9131a115b3074185b3f6509c992ea342',
                                     '41892ee254314a5593e1995143aaa295',
                                     'c91da2f43e6a4e7fb0666a755b43efd7',
                                     'c3fd9173e1584b5a9bcb0e354290c747'],
                               "ARSAC-5": ['48667e9222e04c61ae34b7669b4cb20e',
                                         'c4da0c9a138c44138fe5fec570142634',
                                         'be52113d2b6d428ba403feab07cb1fc0',
                                         '4610a2510eac4c1b89af6b1a5985f949',
                                         '15d705ddef2248cea9bfc48a3a2a848a']}

humanoid_walk_base_dict4 = {"SAC": ['d2014eeb19034c1b89187ba315ed6851',
                                    '1a33932b508240a1ae11289bcefba9f0',
                                    '834dc29681b74705bfc7328321f8d33c',
                                    '6dcc45fb34694a43a77d261c220359b3',
                                    'b278744d70c44d238369aa1c8591658e'],
                            "ARSAC-3": ['b223d148021e444885db36c8212ab91a',
                                         '53c353ecb11e4fdb97ff87481c2bef90',
                                         '159eade496f246e5b51fa0ee7c864644',
                                         'f1a095308fe149d5a33f688067861881',
                                         '20661ba090e2422586618cf543a8db00'],
                               "ARSAC-5": ['77b0e57f24ce419baec3414030ca9b29',
                                         '8209c120f6074a9bb23ddd11b475726f',
                                         '3dc93883ee494821bbf0a32487e49df6',
                                         '77b0e57f24ce419baec3414030ca9b29',
                                         '21a9cc1dba8543d0b147c5dd11d9cc37'],
                            "ARSAC-10": ['12bcc10a1a7642248a876e179eb4cd26',
                                         'cf80e38223a74070a8d25784af6144a1',
                                         '74a0d413956e411484dd15529ebbed58',
                                         'ab715a319ad54c64a3cdcf1a338022f3',
                                         '53c353ecb11e4fdb97ff87481c2bef90']}

humanoid_run_base_dict4 = {"SAC": ['d4f897346e714022bb2e7cc3e1b795b7',
                                   'd4d0f5333ec84290bdc3342885be90f6',
                                   '4ccab64dd972429892e79b2b80ef9ae1',
                                   '534810614ac54afc9d3c65c2f42b5f79',
                                   '4f5781e5784e402c9176eea4ee0e20ef'],
                            "ARSAC-3": ['7a22d3a7eaae48d9bb39441e3fa7a0ba',
                                         '85e6c9a14c594273938f4522107c3937',
                                         '4eb8a953dd0b4bad95f8f9cd3ad80087',
                                         'af8d597642344d70b2eee24e5c3e9153',
                                         'c4ff509870c8468bab08ad4f178dcb1f'],
                            "ARSAC-5": ['0b22c9785f0440ec9166b99e6aa0ede4',
                                       '51a0de4aaec04d2aa2f5e84eaf89374b',
                                       '0f02afcac7a14116a57e17fbe86f41e3',
                                       '004f7b3ab7c74188a3950ad5526c3a4b',
                                       '51a0de4aaec04d2aa2f5e84eaf89374b'],
                            "ARSAC-10": ['6f69419843de435fa1f08e673b4e627f',
                                         'fd1f00c497834716a53c5bd76dea70cb',
                                         '4ff0e447a50842e1b835a01aa74d20d9',
                                         '71db7e4445464aad8f36dbe9c91ff07a',
                                         '04a5ad091ac54904bbd6139f3ca9b701']
                           }

# Base Tests With 1x32 HS AutoEntropy Tuning, BS 256
walker_walk_base_dict5 = {
    "SAC": ["4a4049ef6cd64db1b9efe21d30f15f40",
            "b7247a4fbbc54691ac08ee08aaceb76b",
            "a48c59871a8f416491694a038a4df0e3",
            "b20abe2ce6fe4c23b14b659c0659737a",
            "d1cbda6a1f4e4f83be480832409d4efe"],
    "ARSAC": ["5691959d6b01421d8dc1b78aaa3937ff",
              "2d26d47b74554c8ca91c7302c616403b",
              "f439a152dc6d462596fababa20323359",
              "756d6844ce8347819f6f6849893c1825",
              "bfee66cd9d6d4a1baced9aecde015bd4"]
}

walker_run_base_dict5 = {
    "SAC" : ["3193d027636e4593a8ba71a84ee21638",
             "6528837f8fc94920a472fe827f98ca63",
             "8e8450864b264fe18ed309a9e6fc6866",
             "b6e6f26e60794ce69943c75138c7731c",
             "29af5554dec04009a3e56a581dd420a8"],
    "ARSAC" : ["74e14df64e214d8ab68fcdb91bd44cb8",
               "1cf5297adce049699f860e1438707bab",
               "5d1f3bea197a4530b8cf22ceb8d44f0f",
               "deb002867c1747b39e62bbbf182b8e68",
               "ed83960ca38e4807be3a659caa9302e5"]
}

hopper_stand_base_dict5 = {
    "SAC": ["732e12e250b644d79e02b74871e55daa",
            "c8b7b7edd9eb498e92127a4a180d2813",
            "8e7b076012ed4035aeb505079cb60995",
            "673e8e7765524e5890dabb0abc49f9e8",
            "f8ed84b52dd44a5ea2b48097d574c519"],
    "ARSAC": ["bb2b4fc7de334f94819a634b59def873",
              "6047c0d508884c9cad19abeb0daf1701",
              "e0c4391944d24506afe99c631ea60bfb",
              "c2698c94780d4a4ab0018a6bd13e6303",
              "d260e3a97ef44b5bac92deb500e3de0b"]
}

hopper_hop_base_dict5 = {
    "SAC": ["b0713cbd76e84f138a8eab35c76e4df0",
            "2320644ff44e4869be45a6ba6df06538",
            "5aa24eace5e3442383092e29dc2d23dc",
            "ce57c25d3bf349c2aaf850e383e1359a",
            "36f69828aa1e4d869074179a0831c947"],
    "ARSAC": ["aeeb746726ef4b07a573a98df0711f9c",
              "4ffdd143801a4db4800efe5063ce95df",
              "7f689974083d43869bf1a8240c96e939",
              "037d89a28af84ac2aabf170e46c8cb93",
              "df6ce5eefc2e4e5d8a9d126e17da481a"]
}

quadruped_walk_base_dict5 = {
    "SAC" : ["f1d1b694ebf74bd1a7afae1d27fe3687",
             "5210201af61e43938daafefb77affc43",
             "fbf5904caed6479f8734cd20bc90199e",
             "8532de8d1d1947b7b9946a0b6b927bc5",
             "f91f1ab10909466aa63d5ec9bc3ce5b6"],
    "ARSAC" : ["14ae9513a53a42bf89b65d02f6cdc5e7",
               "8810a8f56bba4a7a878e1b5e51f92098",
               "691c1a88efff4befb0f874b8321d94ef",
               "083935a6c23f4e8685f49e5835f6b7e7",
               "e46ec218807342caa98989cd31379182"]
}

quadruped_run_base_dict5 = {
    "SAC" : ["0eeb188dba754404a41ed786a52d5ac8",
             "55bb145d0b4b4921a60a154504160e0b",
             "cc69bfb3e07d433dbe49977dc4a34b79",
             "2e1b579760ac4ff68adf3a4e11181803",
             "4f295c37b35a4269a016390cd522ffe3"],
    "ARSAC" : ["56911a04f19347898de823f159cf0f52",
               "41b6b5e0401945cfb26f6bbe201720a3",
               "bb9705e69aad45cd85395eefb1f267b0",
               "a56a0c85760c466bb24daa092041a4fe",
               "674eca50665948a089295335e8de13b0"]
}

cheetah_run_base_dict5 = {
    "SAC": ["82891be32c8f4ffb850f4beb1a6c7934",
            "9b30bd559e83454587a5d29e5129a96c",
            "2ed80219827d472eb3151ff204339886",
            "3dd6d13174fb40349b8e2fecf64a5b81",
            "63585d83dafb49b99316d6814b4bbe03"],
    "ARSAC": ["5ab462af89fd4f3ca670838c252d8dc8",
              "15391fe3d8d6428b885d8f748e5e91d8",
              "a0120de87a9541fd968c32812eb02e6a",
              "72b7d63ccfd24675b5a5c14aaa94c2fa",
              "2f23146b1baa45c8806a081cbfb9023e"]
}

# DM Control Pixel Tests
walker_walk_pixel_dict = {"SAC": ['6392d6c1f77547429ab16c46d20f339c',
                                    '',
                                    ''],
                           "ARSAC": ['2c27e50c296e44509e3d7ca60387d130',
                                     '',
                                     '']}

to_plot_transfer_dict2 = {
    "Walker Run Transfer AutoEnt 2x256HS" : walker_transfer_dict2,
    "Quadruped Run Transfer AutoEnt 2x256HS" : quadruped_transfer_dict2,
    "Hopper Hop Transfer AutoEnt 2x256HS" : hopper_transfer_dict2
}

to_plot_transfer_dict3 = {
    "Walker Run Transfer AutoEnt 1x32HS" : walker_transfer_dict3,
    "Quadruped Run Transfer AutoEnt 1x32HS" : quadruped_transfer_dict3,
    #"Hopper Hop Transfer AutoEnt 1x32HS" : hopper_transfer_dict3
}

to_plot_dict = {
        # "Walker Run Transfer" : walker_transfer_dict,
        # "Walker Walk Base" : walker_walk_base_dict,
        # "Walker Run Base" : walker_run_base_dict,
        # "Quadruped Walk Base" : quadruped_walk_base_dict,
        # "Quadruped Run Base" : quadruped_run_base_dict,
        # "Hopper Stand Base" : hopper_stand_base_dict,
        # "Hopper Hop Base" : hopper_hop_base_dict
}

to_plot_dict2 = {
        "Walker Walk 256BS 1x32HS" : walker_walk_base_dict2,
        "Walker Run 256BS 1x32HS" : walker_run_base_dict2,
        "Quadruped Walk 256BS 1x32HS" : quadruped_walk_base_dict2,
        "Quadruped Run 256BS 1x32HS" : quadruped_run_base_dict2,
        "Hopper Stand 256BS 1x32HS" : hopper_stand_base_dict2,
        "Hopper Hop 256BS 1x32HS" : hopper_hop_base_dict2
    }

to_plot_dict3 = {
    "Quadruped Walk 256BS 2x256HS" : quadruped_walk_base_dict3,
    "Quadruped Run 256BS 2x256HS" : quadruped_run_base_dict3,
    "Hopper Stand 256BS 2x256HS" : hopper_stand_base_dict3,
    "Hopper Hop 256BS 2x256HS" : hopper_hop_base_dict3
}

to_plot_dict4 = {
    # "Walker Walk AutoEnt 256BS 2x256HS" : walker_walk_base_dict4,
    # "Walker Run AutoEnt 256BS 2x256HS" : walker_run_base_dict4,
    # "Quadruped Walk AutoEnt 256BS 2x256HS" : quadruped_walk_base_dict4,
    # "Quadruped Run AutoEnt 256BS 2x256HS" : quadruped_run_base_dict4,
    # "Hopper Stand AutoEnt 256BS 2x256HS" : hopper_stand_base_dict4,
    # "Hopper Hop AutoEnt 256BS 2x256HS" : hopper_hop_base_dict4,
    # "Cheetah Run AutoEnt 256BS 2x256HS" : cheetah_run_base_dict4,
    # "Swimmer Swimmer6 AutoEnt 256BS 2x256HS" : swimmer_swimmer6_base_dict4,
    # "Humanoid Stand Autoent 256BS 2x256HS" : humanoid_stand_base_dict4,
    # "Humanoid Walk AutoEnt256BS 2x256HS" : humanoid_walk_base_dict4,
    "Humanoid Run AutoEnt 256BS 2x256HS" : humanoid_run_base_dict4
}

to_plot_dict5 = {
    "Walker Walk AutoEnt 256BS 1x32HS" : walker_walk_base_dict5,
    "Walker Run AutoEnt 256BS 1x32HS" : walker_run_base_dict5,
    "Hopper Stand AutoEnt 256BS 1x32 HS" : hopper_stand_base_dict5,
    "Hopper Hop AutoEnt 256BS 1x32 HS" : hopper_hop_base_dict5,
    "Quadruped Walk AutoEnt 256BS 1x32HS" : quadruped_walk_base_dict5,
    "Quadruped Run AutoEnt 256BS 1x32HS" : quadruped_run_base_dict5,
    "Cheetah Run AutoEnt 256BS 1x32 HS" : cheetah_run_base_dict5
}

if __name__ == "__main__":
    # Specify the folder we want to save the visualizations to
    base_rew_dir = "reward_plots_new/"
    base_logscale_dir = "log_scale_plots_new/"
    for env in to_plot_transfer_dict3.keys():
        env_exp_dict = to_plot_transfer_dict3[env]
        print("Visualizing ", env)
        plot_rewards(env_exp_dict, base_rew_dir, REWARD_KEY)
        plot_rewards(env_exp_dict, base_logscale_dir, LOG_SCALE_KEY)