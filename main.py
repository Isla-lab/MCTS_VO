import argparse
import gc
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from numpy import mean, std

from bettergym.agents.planner_mcts import Mcts
from bettergym.agents.planner_mcts_apw import MctsApw
from bettergym.agents.utils.utils import voo, towards_goal, uniform_towards_goal
from bettergym.agents.utils.vo import sample_centered_robot_arena, voo_vo, towards_goal_vo, uniform_random_vo
from environment_creator import create_env_five_small_obs_continuous, create_env_four_obs_difficult_continuous
from experiment_utils import print_and_notify, plot_frame, create_animation_tree_trajectory
from mcts_utils import uniform_random

DEBUG_DATA = True
DEBUG_ANIMATION = True
ANIMATION = True


@dataclass(frozen=True)
class ExperimentData:
    rollout_policy: Callable
    action_expansion_policy: Callable
    discrete: bool
    obstacle_reward: bool
    std_angle: float
    n_sim: int = 1000
    c: float = 150


@njit
def seed_numba(seed_value: int):
    np.random.seed(seed_value)


def set_random_state():
    random_s = (3, (
    2599973022, 3693581950, 3434803887, 772706845, 1938947815, 2261994492, 3734151276, 3773337150, 167350229,
    1617671557, 1710716744, 1785225396, 121834244, 2390916739, 2178681923, 2526620286, 102400407, 511533624, 713599401,
    2033711583, 2757453820, 2884527182, 3892792597, 3455186386, 2568697814, 2392001036, 2417328795, 503956998,
    2297969920, 4227817457, 272493779, 2135761731, 3572621217, 2128523595, 4240098396, 1248414502, 27087287, 3944215025,
    2514264267, 1594509742, 1888391299, 350672206, 1275154943, 2130672201, 3364600663, 2273530238, 3258119695,
    3130015288, 3508015414, 3293133216, 47594031, 3945168292, 2335949959, 3163124740, 937761415, 218262679, 4112662065,
    4036565289, 62002191, 3061719350, 196119323, 1621366404, 458676989, 2874392817, 2219420612, 151012214, 3064725259,
    3073281243, 4250534333, 1260211258, 3057189403, 59627953, 2335647561, 25184509, 2612882507, 544853548, 1874718702,
    794052680, 529713919, 2238170673, 856057840, 2679329338, 2898088159, 3738270284, 2407890582, 2558461664, 2104858016,
    2483746841, 1324780134, 1106041847, 1875633777, 4229192381, 1502584482, 2212548448, 2346084482, 72329928,
    2749587371, 1860144796, 293330490, 3886414281, 288776256, 3360370788, 2504065507, 3611087241, 3311062508,
    3986018491, 2723276828, 1918860186, 2501454717, 2240075044, 1268637723, 571104873, 540694380, 2291713845,
    1639948253, 4192192124, 4173079261, 2200038948, 1424476286, 1048705227, 1410089676, 2491128856, 115569867,
    209473227, 1850074413, 2534878694, 4270119998, 2094479992, 777987279, 4233355250, 2480660846, 725514424, 3213176188,
    314580940, 2685679300, 1581790957, 2509502400, 506791059, 753922853, 1995370248, 688455822, 478479836, 1296401440,
    878022930, 3723272198, 3861713005, 1429771866, 2913205883, 2331125969, 717936327, 3132082624, 3302014928,
    3690047410, 2522425178, 1737916514, 398152474, 3373717903, 3286769813, 163742690, 3979846518, 3708463475, 706642042,
    2932160025, 3588424682, 1094610024, 4237701117, 295004280, 2787793357, 998765051, 1213004830, 3541034923,
    4068713394, 680696954, 2314397548, 1804305268, 2270676795, 1526152121, 1152889495, 4140528789, 3395394361,
    1220206091, 3721287458, 2743347512, 2057026222, 125639038, 2880097157, 3923989731, 305875655, 2409960989,
    2408003379, 2126796826, 1268663821, 4030518392, 2127670523, 470371888, 203100617, 1010959782, 1084057215,
    3654512434, 323404555, 3770377634, 844503248, 55249466, 3696240474, 2692964319, 1543056097, 4076766527, 968799855,
    1189660566, 3212575478, 1461822938, 3381269365, 3635348405, 2547428942, 4169301629, 1076651551, 2292668094,
    3354213221, 3836469524, 1868858068, 1749598685, 2356939243, 3490072030, 356284844, 3786798597, 3026487971,
    3433306435, 1205699481, 1900800010, 3677890771, 1730809837, 3199001099, 3534161647, 2773068693, 4069267244,
    3181289772, 160962462, 4263026152, 1794117149, 1706211875, 2195358645, 409791915, 3356310201, 1054408810,
    2746164955, 1515854265, 2913425889, 2646576012, 2896640036, 1951937474, 3315511534, 809703847, 170086134,
    3677959259, 2519811637, 814495529, 939629267, 3183156321, 137668573, 767695039, 2829212193, 3168312567, 2018655572,
    1774001845, 402464107, 2157667298, 3794972536, 2104779687, 3437184531, 3437424407, 2799245660, 3027263150,
    2489571189, 3309495831, 3421983099, 1047898072, 2011390357, 304609557, 174452608, 4012195093, 1972864253,
    3915266213, 705423826, 2538656534, 1501004734, 963154421, 3552540849, 4130655474, 3284618671, 1781352069,
    2089228068, 2242948523, 2881472168, 1205983517, 687015029, 614316120, 3153488599, 2490556024, 3840968001,
    2314348234, 3745253284, 179058162, 2393716324, 854178498, 2761546782, 797931100, 282967441, 409162359, 1947728074,
    1322853092, 332821501, 1938537295, 2181440929, 2408100647, 3546707744, 4208971565, 1691003913, 3396351296,
    2837758564, 2313936017, 2053986091, 2718067764, 1413329338, 2775640104, 263520030, 160405901, 539642979, 3037343479,
    2069067769, 3056222850, 1660274454, 3543021912, 3375812523, 3439740243, 2980810442, 1305017031, 955307606,
    2164982325, 1000358276, 197706289, 1280442117, 445048506, 3360287068, 3813089072, 3552289766, 971512164, 3199832044,
    1958928974, 1806975078, 2306170889, 3996840609, 2231164722, 2893759008, 3639898695, 4277313951, 4232135409,
    1597264588, 3915725934, 2335026852, 2257871938, 3522636016, 2054225181, 268630680, 2607797437, 2794501591,
    344698771, 3268109993, 3997129792, 1065078808, 598280387, 3082169905, 3537693681, 522237845, 4155796162, 383277692,
    2255109713, 4192388718, 1707809146, 3532663406, 2003193452, 4238576160, 2951929611, 2748392547, 3798870485,
    36355073, 3595583945, 4175145842, 1824322195, 2026507270, 172449183, 2302255530, 2990287120, 3687597311, 2942865670,
    1688014298, 49590401, 929077547, 100138653, 3379812098, 902724688, 4138848121, 1120810366, 3937022150, 1871104344,
    3314089350, 4272863927, 675353987, 1213983980, 495536068, 2184711709, 1740894095, 622395464, 1310984052, 3570045692,
    2965812873, 1653397025, 4250916856, 809166028, 2605222133, 1949076122, 4195503686, 131359013, 2762544838, 808915442,
    3199227505, 951237613, 2929402865, 774427875, 4024280727, 1688641742, 3794085337, 4109690111, 3290841619, 68625642,
    3661438332, 2218129377, 1529819618, 1415376521, 1529534264, 3358479499, 644454209, 3098976191, 4260727401,
    250255063, 399754086, 2352084486, 1834719237, 3857779260, 2954895564, 1701358281, 30325275, 2192081385, 632812032,
    2772145234, 1823667992, 137628186, 1242366100, 100222557, 2300414767, 2627619182, 426000932, 2329737391, 1859159456,
    3735361397, 1399068395, 2665678733, 2261631409, 608255722, 196736577, 507397208, 2633119802, 2559542229, 2785463115,
    3624018049, 3332829366, 1189616070, 845708669, 28301418, 2469975319, 3128491946, 388494940, 1670225634, 1408632213,
    3211875359, 253831886, 1549961465, 1880327835, 3306364109, 688945773, 3841969282, 2992788100, 942473132, 4001401281,
    517653791, 205953876, 3474820849, 2880047528, 3687452312, 1520841099, 159965134, 3822379387, 3018908199, 891095680,
    3235273382, 1195753706, 1834382968, 2298919908, 3571377280, 1592377572, 3098768278, 2769228469, 3352220924,
    1343040247, 2539519183, 3657846513, 2711900653, 415519354, 545358302, 4232018496, 1863073356, 4130937839,
    1118495402, 3747782210, 4080979895, 3389765855, 2032005539, 3459243884, 4205225354, 665406157, 2586556937,
    2310726683, 2419467911, 4178967031, 3139362684, 1254827781, 1202845564, 2399827796, 1121780450, 63033591,
    3156467899, 2271242773, 576837577, 719484941, 1539266929, 3249599464, 933989048, 3618297579, 3488650303, 624091705,
    778717147, 1382153525, 3935212772, 751155837, 4162629268, 3614461269, 4279570963, 1731371956, 3916652074,
    1006459357, 1714733678, 931537507, 286390328, 4289362661, 1115477581, 328169119, 2957020140, 3916430190, 788977835,
    707138968, 1069929359, 2652420421, 557482177, 2439379283, 1229184745, 3461041047, 377529944, 3540002830, 2884367652,
    957925651, 212011471, 1078792963, 2907544733, 145735298, 373016536, 2814306287, 1486510253, 1698082439, 2337835437,
    701178909, 3828764491, 2343502579, 4287449752, 3498199531, 698436426, 3612410483, 2467203406, 2282053586,
    3501801185, 3773243726, 1344458916, 2522383345, 4211128091, 3923471734, 2885989263, 2260736926, 1376839816,
    2014163109, 1080910093, 874731936, 2202115139, 3657005512, 3622156099, 4180139310, 3770142, 3912583617, 3837342049,
    2988738698, 4211581764, 3387728904, 2109459683, 2234211671, 967159371, 1094941523, 974954852, 584), None)
    np_s = ('MT19937', np.array([1826541397, 3919491871, 582002114, 1072002853, 3427816492,
                                 3277710540, 2190536751, 2376824353, 215931097, 1376495175,
                                 2878224986, 4262691606, 1774530346, 1054727364, 1018863849,
                                 3001937663, 107950565, 115668207, 3875809262, 976075454,
                                 929408565, 3971991713, 1257747764, 158793202, 3283119217,
                                 3680225154, 1157445090, 486338703, 183047369, 2976616662,
                                 3822767414, 2766777485, 4245040028, 1378067321, 1789303079,
                                 461614754, 2851584708, 4007063474, 2853325531, 470276027,
                                 1694538863, 2853547363, 2592400295, 967669129, 2919711882,
                                 1711824496, 1255800359, 2237665018, 2892820054, 2148501578,
                                 574856618, 179300077, 761416795, 621768415, 2086020440,
                                 265787885, 3172414035, 2713901845, 4085929262, 3437281534,
                                 2733891714, 2903478520, 3931239907, 3622869204, 783790034,
                                 3428209745, 3437820397, 1907432065, 4110609805, 4168555546,
                                 2723098662, 3533887743, 2484469858, 646441705, 2224080658,
                                 1770789535, 1496972845, 2525913347, 1532743945, 4049351324,
                                 1185654136, 1934650698, 559114476, 4057056787, 3989736119,
                                 918951280, 397176143, 2900100301, 1617223158, 2547436794,
                                 196131027, 2152640491, 2620006611, 3383783181, 1242159538,
                                 2544116793, 4288600047, 2156761304, 367356787, 3328258367,
                                 1862536675, 820368978, 2136416067, 3998202948, 2589879725,
                                 488113269, 3533616230, 1311524508, 3638551842, 4257014881,
                                 3561173344, 1963015037, 2644304077, 3361303524, 3248738701,
                                 2992932244, 2187758668, 866612970, 3229032027, 3199136939,
                                 3876648918, 464507120, 2630177507, 4124857204, 1260596254,
                                 1667585914, 3070517605, 4243011432, 1127681403, 1321531130,
                                 3962167039, 4172058070, 4271572097, 3321534353, 3990346382,
                                 1047038037, 2329187286, 3148539708, 44895485, 432393308,
                                 2826080241, 1768794444, 2277885427, 889418695, 1341309443,
                                 770393796, 3569496288, 202420887, 1330572428, 4161900056,
                                 4049079779, 2326847896, 2865303460, 3354501373, 169951776,
                                 1537321819, 30420911, 2340055956, 4023485697, 214938222,
                                 2006691433, 1618462619, 4103899426, 2130289995, 4094001233,
                                 518500331, 691447059, 3325572915, 3078142293, 3111024043,
                                 2671215728, 1600987384, 3473103510, 1468498504, 1469370617,
                                 1891651152, 1505274009, 3487220377, 152942729, 2896932853,
                                 556770201, 3579432448, 2265107603, 1805893083, 3587583929,
                                 667769200, 468012355, 2669851356, 2079798326, 3622401033,
                                 1957520909, 1778339705, 3380364874, 59459229, 2229564000,
                                 4169324781, 216249261, 3701641800, 330685716, 194222528,
                                 3187714363, 2279603685, 702641630, 3249746846, 918912088,
                                 1276491975, 1754116608, 2342563476, 1321182711, 3851229521,
                                 1838650818, 4209193681, 855273535, 3921643071, 54542842,
                                 208593788, 784021141, 1514963618, 3237462723, 1327430162,
                                 3921585856, 3191527289, 859833522, 2410725429, 2117623047,
                                 1081492533, 2623815942, 1691651922, 2159394997, 310440951,
                                 1678309301, 2665501224, 227965630, 246023593, 2612220163,
                                 1822751972, 341793312, 394341510, 1425596302, 1465038316,
                                 1217502048, 2166815696, 1325139472, 4029270833, 4203965573,
                                 3731565790, 2063168550, 710142524, 2704041408, 1743567092,
                                 2289776591, 3475708213, 1821670846, 3718065710, 711691497,
                                 4112738846, 4010237705, 2112158977, 3428654511, 3107187854,
                                 1756531756, 2654456001, 63990953, 2595395159, 4016450309,
                                 2981884130, 3363322181, 886351951, 2825706901, 2938483256,
                                 2557024749, 1314168676, 2048254694, 1753187019, 3599373425,
                                 758244380, 1395506695, 3253390315, 933512138, 1470774395,
                                 724099714, 2693366592, 91210213, 299312464, 2597404455,
                                 1593690815, 2224963355, 2995416198, 2647543516, 928478809,
                                 1034367727, 1977179429, 2870646973, 3250331279, 628192044,
                                 3718415125, 1867277152, 21079702, 3164113069, 756011449,
                                 2368814491, 3740085159, 3663643919, 2664806653, 3399998989,
                                 2904927917, 1756712691, 1674471864, 324955378, 4188526034,
                                 1939951653, 1933823279, 3818278646, 3315099703, 2725923615,
                                 559244196, 829245261, 3968369370, 2177566534, 144293052,
                                 2238785946, 1961004199, 3002629099, 4178460787, 2438423094,
                                 184361984, 3154743703, 1839094166, 416159792, 946067181,
                                 3262561550, 1921477245, 3114795680, 1277760707, 2714081391,
                                 3776566921, 3959508945, 3260900470, 3682359862, 240585335,
                                 1069286230, 1172508403, 2023963335, 621461756, 2297807760,
                                 1502120347, 4060968699, 191156271, 2375547475, 424517480,
                                 143642158, 1459435415, 1291872408, 2751283430, 1604647401,
                                 2782170187, 3093782963, 2588592664, 2581239778, 3239274776,
                                 4196996901, 69629487, 685647457, 2423251453, 2448287643,
                                 2043804528, 3332907088, 614559410, 3721228565, 987785555,
                                 2018105748, 926933403, 383785925, 2275114396, 3671278171,
                                 2163110887, 268585945, 2821194704, 3669782361, 3689416042,
                                 70672469, 3611721358, 2809371501, 3229136233, 768113052,
                                 3735599004, 1221928232, 1167484786, 2324718854, 3967274666,
                                 3640191098, 3464008831, 676166263, 3757122927, 613034294,
                                 2382131852, 3131333660, 2790780532, 1640297702, 274073196,
                                 3239337346, 1030496582, 1830697618, 2526538023, 2326313203,
                                 3740248171, 2404550052, 2266709539, 194163135, 3752582089,
                                 2098811404, 2384707872, 3065488934, 1582499781, 3509908587,
                                 4218238690, 2864494337, 4234168013, 3322056365, 2592125368,
                                 305886008, 2574288893, 4257449874, 2226597018, 3681151056,
                                 1506815175, 4216859918, 756416160, 1458903273, 3417998202,
                                 516186907, 2434883598, 3246538621, 3742661954, 737669607,
                                 353376695, 1188022014, 2638261739, 383543093, 2203429044,
                                 2166410714, 1850923882, 685896395, 841284278, 1588906347,
                                 2051033069, 3774673634, 1932314123, 3151058805, 453785995,
                                 1775802028, 277844016, 3737986891, 4217070608, 2940378124,
                                 1061551708, 660827986, 2993773850, 3135960773, 2016766553,
                                 2093609086, 1116156064, 874494011, 239832650, 114808194,
                                 3898379564, 358266011, 1373992821, 2234803694, 847852798,
                                 1659544728, 2727812793, 1834312266, 2205662255, 2788174350,
                                 553005580, 45773960, 86057702, 1824180576, 1700087198,
                                 978297620, 348826987, 1377875765, 3790567575, 4079057024,
                                 3407415661, 3845777254, 2884432660, 3969831293, 1315655924,
                                 3237737889, 3946605858, 3898376026, 3273459975, 3632239454,
                                 3518986095, 3080961787, 1464191558, 1291973629, 2756752020,
                                 1531450372, 2498784254, 2082784444, 2311208167, 2271491708,
                                 2705296472, 3601275399, 3472146806, 1623642613, 374749500,
                                 2136943009, 3794629639, 1624557164, 1538389019, 3295790084,
                                 956999015, 2841509773, 4167734426, 448111890, 2692166245,
                                 1219748759, 3291900527, 856202300, 1819317901, 4032269844,
                                 361403738, 3682698065, 1720704785, 1050146312, 2674315337,
                                 100528104, 1154643800, 2108902187, 1194812697, 1313877472,
                                 1780984133, 2723512416, 412056106, 2409244888, 2905546748,
                                 312084246, 1541539454, 3413578384, 1442480634, 3520152956,
                                 2762024668, 3879824796, 4026581109, 2593576809, 1381562685,
                                 132120089, 3138660940, 3017751013, 3307272718, 327226020,
                                 926896921, 3327595682, 680121870, 2448956814, 1418713509,
                                 4210829511, 1232248842, 1081514090, 2135859831, 2191004590,
                                 2230196428, 1505705965, 561694950, 3469614280, 4022089693,
                                 977335385, 2554599331, 3885140044, 776062397, 2547975304,
                                 103246868, 61009885, 309611225, 3173071147, 2193830915,
                                 671205915, 3869859124, 360004357, 382353112, 1866425043,
                                 152147151, 350182584, 2715000933, 102272466, 4130489396,
                                 2101486174, 730475374, 682963480, 3070999643, 3486537135,
                                 2961717328, 632496380, 4130445062, 3110029452, 911797310,
                                 260732290, 3806728688, 2923688649, 3173154010, 3207528210,
                                 540869659, 3391745199, 3948633572, 2661565663, 864665808,
                                 3454032684, 3107990357, 769832800, 3803443969, 4254185964,
                                 1585553652, 193535425, 35599795, 4293894106, 1778308333,
                                 1628386430, 608597932, 4233006682, 1438878569], dtype=np.uint32), 502, 0, 0.0)
    random.setstate(random_s)
    np.random.set_state(np_s)


def get_random_state():
    print(random.getstate())
    print('*' * 50)
    print(np.random.get_state())


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    seed_numba(seed_value)


def run_experiment(experiment: ExperimentData, arguments):
    global exp_num
    # input [forward speed, yaw_rate]
    real_env, sim_env = create_env_four_obs_difficult_continuous(initial_pos=(1, 1),
                                                                 goal=(2, 10),
                                                                 discrete=experiment.discrete,
                                                                 rwrd_in_sim=experiment.obstacle_reward,
                                                                 out_boundaries_rwrd=arguments.rwrd,
                                                                 dt_sim=arguments.dt,
                                                                 n_vel=arguments.v,
                                                                 n_angles=arguments.a)

    # real_env, sim_env = create_env_five_small_obs_continuous(initial_pos=(1, 1),
    #                                                          goal=(10, 10),
    #                                                          discrete=experiment.discrete,
    #                                                          rwrd_in_sim=experiment.obstacle_reward,
    #                                                          out_boundaries_rwrd=arguments.rwrd,
    #                                                          dt_sim=arguments.dt,
    #                                                          n_vel=arguments.v,
    #                                                          n_angles=arguments.a)
    s0, _ = real_env.reset()
    trajectory = np.array(s0.x)
    config = real_env.config

    goal = s0.goal

    s = s0
    # set_random_state()
    # real_env.gym_env.state.x = np.array([2.01604743, 0.01678031, -0.19841203, 0.07777778])
    # sim_env.gym_env.state.x = real_env.gym_env.state.x.copy()

    if experiment.action_expansion_policy is not voo_vo:
        for o in s0.obstacles:
            o.radius *= 1.05
        real_env.gym_env.state = s0
        sim_env.gym_env.state = s0

    obs = [s0.obstacles]
    if not experiment.discrete:
        planner = MctsApw(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env,
            computational_budget=100,
            k=arguments.k,
            alpha=arguments.alpha,
            discount=0.99,
            action_expansion_function=experiment.action_expansion_policy,
            rollout_policy=experiment.rollout_policy
        )
    else:
        planner = Mcts(
            num_sim=experiment.n_sim,
            c=experiment.c,
            environment=sim_env,
            computational_budget=100,
            discount=0.99,
            rollout_policy=experiment.rollout_policy
        )
    print("Simulation Started")
    terminal = False
    rewards = []
    times = []
    infos = []
    actions = []
    step_n = 0
    while not terminal:
        step_n += 1
        if step_n == 1000:
            break
        print(f"Step Number {step_n}")
        initial_time = time.time()
        # if step_n == 123:
        #     print(s.x)
        #     get_random_state()
        u, info = planner.plan(s)
        # del info['q_values']
        # del info['actions']
        # del info['visits']
        # gc.collect()

        actions.append(u)
        min_angle = s.x[2] - 1.9 * config.dt
        max_angle = s.x[2] + 1.9 * config.dt
        u_copy = np.array(u, copy=True)
        u_copy[1] = max(min(u_copy[1], max_angle), min_angle)
        u_copy[1] = (u_copy[1] + math.pi) % (2 * math.pi) - math.pi
        final_time = time.time() - initial_time
        # visualize_tree(planner, step_n)
        infos.append(info)

        times.append(final_time)
        s, r, terminal, truncated, env_info = real_env.step(s, u_copy)
        sim_env.gym_env.state = real_env.gym_env.state.copy()
        rewards.append(r)
        trajectory = np.vstack((trajectory, s.x))  # store state history
        obs.append(s.obstacles)
        gc.collect()

    exp_name = '_'.join([k + ':' + str(v) for k, v in arguments.__dict__.items()])
    print_and_notify(
        f"Simulation Ended with Reward: {round(sum(rewards), 2)}\n"
        f"Discrete: {experiment.discrete}\n"
        f"Std Rollout Angle: {experiment.std_angle}\n"
        f"Number of Steps: {step_n}\n"
        f"Avg Reward Step: {round(sum(rewards) / step_n, 2)}\n"
        f"Avg Step Time: {np.round(mean(times), 2)}±{np.round(std(times), 2)}\n"
        f"Total Time: {sum(times)}\n"
        f"Num Simulations: {experiment.n_sim}",
        exp_num,
        exp_name
    )

    data = {
        "cumRwrd": round(sum(rewards), 2),
        "nSteps": step_n,
        "MeanStepTime": np.round(mean(times), 2),
        "StdStepTime": np.round(std(times), 2)
    }
    data = data | arguments.__dict__
    df = pd.Series(data)
    df.to_csv(f'{exp_name}_{exp_num}.csv')

    if ANIMATION:
        print("Creating Gif...")
        fig, ax = plt.subplots()
        ani = FuncAnimation(
            fig,
            plot_frame,
            fargs=(goal, config, obs, trajectory, ax),
            frames=len(trajectory),
            save_count=None,
            cache_frame_data=False
        )
        ani.save(f"debug/trajectory_{exp_name}_{exp_num}.gif", fps=150)
        plt.close(fig)

    trajectories = [i["trajectories"] for i in infos]
    rollout_values = [i["rollout_values"] for i in infos]
    if DEBUG_DATA:
        print("Saving Debug Data...")
        q_vals = [i["q_values"] for i in infos]
        visits = [i["visits"] for i in infos]
        a = [[an.action for an in i["actions"]] for i in infos]
        with open(f"debug/trajectories_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(trajectories, f)
        with open(f"debug/rollout_values_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(rollout_values, f)
        with open(f"debug/visits_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(visits, f)
        with open(f"debug/q_values_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(q_vals, f)
        with open(f"debug/actions_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(a, f)
        with open(f"debug/trajectory_real_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(trajectory, f)
        with open(f"debug/chosen_a_{exp_name}_{exp_num}.pkl", 'wb') as f:
            pickle.dump(actions, f)

    if DEBUG_ANIMATION:
        print("Creating Tree Trajectories Animation...")
        create_animation_tree_trajectory(goal, config, obs, exp_num, exp_name, rollout_values, trajectories)
        # create_animation_tree_trajectory_w_steps(goal, config, obs, exp_num)
    gc.collect()
    print("Done")


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--algorithm', default="vanilla", type=str, help='The algorithm to run')
    parser.add_argument('--nsim', default=1000, type=int, help='The number of simulation the algorithm will run')
    parser.add_argument('--rwrd', default=-100, type=int, help='')
    parser.add_argument('--dt', default=0.2, type=float, help='')
    parser.add_argument('--std', default=0.38 * 2, type=float, help='')
    parser.add_argument('--amplitude', default=1, type=float, help='')
    parser.add_argument('--c', default=1, type=float, help='')
    parser.add_argument('--rollout', default="normal_towards_goal", type=str, help='')
    parser.add_argument('--alpha', default=0.1, type=float, help='')
    parser.add_argument('--k', default=50, type=float, help='')
    parser.add_argument('--a', default=10, type=int, help='number of discretization of angles')
    parser.add_argument('--v', default=10, type=int, help='number of discretization of velocities')
    parser.add_argument('--num', default=1, type=int, help='number of experiments to run')

    return parser


def get_experiment_data(arguments):
    # var_angle = 0.38 * 2
    std_angle_rollout = arguments.std

    if arguments.rollout == "normal_towards_goal":
        if arguments.algorithm == "VO2":
            rollout_policy = partial(towards_goal_vo, std_angle_rollout=std_angle_rollout)
        else:
            rollout_policy = partial(towards_goal, std_angle_rollout=std_angle_rollout)
    elif arguments.rollout == "uniform_towards_goal":
        rollout_policy = partial(uniform_towards_goal, amplitude=math.radians(arguments.amplitude))
    elif arguments.rollout == "uniform":
        if arguments.algorithm == "VO2":
            rollout_policy = uniform_random_vo
        else:
            rollout_policy = uniform_random
    else:
        raise ValueError("rollout function not valid")

    sample_centered = partial(sample_centered_robot_arena, std_angle=std_angle_rollout)
    if arguments.algorithm == "VOR":
        # VORONOI + VO (albero + reward ostacoli)
        return ExperimentData(
            action_expansion_policy=partial(voo_vo, eps=0.3, sample_centered=sample_centered),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
        )
    elif arguments.algorithm == "VOO":
        # VORONOI
        return ExperimentData(
            action_expansion_policy=partial(voo, eps=0.3, sample_centered=sample_centered),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
        )
    elif arguments.algorithm == "VANILLA":
        # VANILLA
        return ExperimentData(
            action_expansion_policy=None,
            rollout_policy=rollout_policy,
            discrete=True,
            obstacle_reward=True,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
        )
    else:
        # VORONOI + VO (albero + rollout)
        return ExperimentData(
            action_expansion_policy=partial(voo_vo, eps=0.3, sample_centered=sample_centered),
            rollout_policy=rollout_policy,
            discrete=False,
            obstacle_reward=False,
            std_angle=std_angle_rollout,
            n_sim=arguments.nsim,
            c=arguments.c
        )


def main():
    global exp_num
    args = argument_parser().parse_args()
    exp = get_experiment_data(args)
    seed_everything(2)
    for exp_num in range(args.num):
        run_experiment(experiment=exp, arguments=args)


if __name__ == '__main__':
    main()
