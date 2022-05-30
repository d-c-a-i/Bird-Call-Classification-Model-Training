import torch
from pathlib import Path

USE_NOCALL = False
SAMPLE_RATE = 32000

BIRD_CODE = {
    'acafly': 0, 'acowoo': 1, 'aldfly': 2, 'ameavo': 3, 'amecro': 4,
    'amegfi': 5, 'amekes': 6, 'amepip': 7, 'amered': 8, 'amerob': 9,
    'amewig': 10, 'amtspa': 11, 'andsol1': 12, 'annhum': 13, 'astfly': 14,
    'azaspi1': 15, 'babwar': 16, 'baleag': 17, 'balori': 18, 'banana': 19,
    'banswa': 20, 'banwre1': 21, 'barant1': 22, 'barswa': 23, 'batpig1': 24,
    'bawswa1': 25, 'bawwar': 26, 'baywre1': 27, 'bbwduc': 28, 'bcnher': 29,
    'belkin1': 30, 'belvir': 31, 'bewwre': 32, 'bkbmag1': 33, 'bkbplo': 34,
    'bkbwar': 35, 'bkcchi': 36, 'bkhgro': 37, 'bkmtou1': 38, 'bknsti': 39,
    'blbgra1': 40, 'blbthr1': 41, 'blcjay1': 42, 'blctan1': 43, 'blhpar1': 44,
    'blkpho': 45, 'blsspa1': 46, 'blugrb1': 47, 'blujay': 48, 'bncfly': 49,
    'bnhcow': 50, 'bobfly1': 51, 'bongul': 52, 'botgra': 53, 'brbmot1': 54,
    'brbsol1': 55, 'brcvir1': 56, 'brebla': 57, 'brncre': 58, 'brnjay': 59,
    'brnthr': 60, 'brratt1': 61, 'brwhaw': 62, 'brwpar1': 63, 'btbwar': 64,
    'btnwar': 65, 'btywar': 66, 'bucmot2': 67, 'buggna': 68, 'bugtan': 69,
    'buhvir': 70, 'bulori': 71, 'burwar1': 72, 'bushti': 73, 'butsal1': 74,
    'buwtea': 75, 'cacgoo1': 76, 'cacwre': 77, 'calqua': 78, 'caltow': 79,
    'cangoo': 80, 'canwar': 81, 'carchi': 82, 'carwre': 83, 'casfin': 84,
    'caskin': 85, 'caster1': 86, 'casvir': 87, 'categr': 88, 'ccbfin': 89,
    'cedwax': 90, 'chbant1': 91, 'chbchi': 92, 'chbwre1': 93, 'chcant2': 94,
    'chispa': 95, 'chswar': 96, 'cinfly2': 97, 'clanut': 98, 'clcrob': 99,
    'cliswa': 100, 'cobtan1': 101, 'cocwoo1': 102, 'cogdov': 103, 'colcha1': 104,
    'coltro1': 105, 'comgol': 106, 'comgra': 107, 'comloo': 108, 'commer': 109,
    'compau': 110, 'compot1': 111, 'comrav': 112, 'comyel': 113, 'coohaw': 114,
    'cotfly1': 115, 'cowscj1': 116, 'cregua1': 117, 'creoro1': 118, 'crfpar': 119,
    'cubthr': 120, 'daejun': 121, 'dowwoo': 122, 'ducfly': 123, 'dusfly': 124,
    'easblu': 125, 'easkin': 126, 'easmea': 127, 'easpho': 128, 'eastow': 129,
    'eawpew': 130, 'eletro': 131, 'eucdov': 132, 'eursta': 133, 'fepowl': 134,
    'fiespa': 135, 'flrtan1': 136, 'foxspa': 137, 'gadwal': 138, 'gamqua': 139,
    'gartro1': 140, 'gbbgul': 141, 'gbwwre1': 142, 'gcrwar': 143, 'gilwoo': 144,
    'gnttow': 145, 'gnwtea': 146, 'gocfly1': 147, 'gockin': 148, 'gocspa': 149,
    'goftyr1': 150, 'gohque1': 151, 'goowoo1': 152, 'grasal1': 153, 'grbani': 154,
    'grbher3': 155, 'grcfly': 156, 'greegr': 157, 'grekis': 158, 'grepew': 159,
    'grethr1': 160, 'gretin1': 161, 'greyel': 162, 'grhcha1': 163, 'grhowl': 164,
    'grnher': 165, 'grnjay': 166, 'grtgra': 167, 'grycat': 168, 'gryhaw2': 169,
    'gwfgoo': 170, 'haiwoo': 171, 'heptan': 172, 'hergul': 173, 'herthr': 174,
    'herwar': 175, 'higmot1': 176, 'hofwoo1': 177, 'houfin': 178, 'houspa': 179,
    'houwre': 180, 'hutvir': 181, 'incdov': 182, 'indbun': 183, 'kebtou1': 184,
    'killde': 185, 'labwoo': 186, 'larspa': 187, 'laufal1': 188, 'laugul': 189,
    'lazbun': 190, 'leafly': 191, 'leasan': 192, 'lesgol': 193, 'lesgre1': 194,
    'lesvio1': 195, 'linspa': 196, 'linwoo1': 197, 'littin1': 198, 'lobdow': 199,
    'lobgna5': 200, 'logshr': 201, 'lotduc': 202, 'lotman1': 203, 'lucwar': 204,
    'macwar': 205, 'magwar': 206, 'mallar3': 207, 'marwre': 208, 'mastro1': 209,
    'meapar': 210, 'melbla1': 211, 'monoro1': 212, 'mouchi': 213, 'moudov': 214,
    'mouela1': 215, 'mouqua': 216, 'mouwar': 217, 'mutswa': 218, 'naswar': 219,
    'norcar': 220, 'norfli': 221, 'normoc': 222, 'norpar': 223, 'norsho': 224,
    'norwat': 225, 'nrwswa': 226, 'nutwoo': 227, 'oaktit': 228, 'obnthr1': 229,
    'ocbfly1': 230, 'oliwoo1': 231, 'olsfly': 232, 'orbeup1': 233, 'orbspa1': 234,
    'orcpar': 235, 'orcwar': 236, 'orfpar': 237, 'osprey': 238, 'ovenbi1': 239,
    'pabspi1': 240, 'paltan1': 241, 'palwar': 242, 'pasfly': 243, 'pavpig2': 244,
    'phivir': 245, 'pibgre': 246, 'pilwoo': 247, 'pinsis': 248, 'pirfly1': 249,
    'plawre1': 250, 'plaxen1': 251, 'plsvir': 252, 'plupig2': 253, 'prowar': 254,
    'purfin': 255, 'purgal2': 256, 'putfru1': 257, 'pygnut': 258, 'rawwre1': 259,
    'rcatan1': 260, 'rebnut': 261, 'rebsap': 262, 'rebwoo': 263, 'redcro': 264,
    'reevir1': 265, 'rehbar1': 266, 'relpar': 267, 'reshaw': 268, 'rethaw': 269,
    'rewbla': 270, 'ribgul': 271, 'rinkin1': 272, 'roahaw': 273, 'robgro': 274,
    'rocpig': 275, 'rotbec': 276, 'royter1': 277, 'rthhum': 278, 'rtlhum': 279,
    'ruboro1': 280, 'rubpep1': 281, 'rubrob': 282, 'rubwre1': 283, 'ruckin': 284,
    'rucspa1': 285, 'rucwar': 286, 'rucwar1': 287, 'rudpig': 288, 'rudtur': 289,
    'rufhum': 290, 'rugdov': 291, 'rumfly1': 292, 'runwre1': 293, 'rutjac1': 294,
    'saffin': 295, 'sancra': 296, 'sander': 297, 'savspa': 298, 'saypho': 299,
    'scamac1': 300, 'scatan': 301, 'scbwre1': 302, 'scptyr1': 303, 'scrtan1': 304,
    'semplo': 305, 'shicow': 306, 'sibtan2': 307, 'sinwre1': 308, 'sltred': 309,
    'smbani': 310, 'snogoo': 311, 'sobtyr1': 312, 'socfly1': 313, 'solsan': 314,
    'sonspa': 315, 'soulap1': 316, 'sposan': 317, 'spotow': 318, 'spvear1': 319,
    'squcuc1': 320, 'stbori': 321, 'stejay': 322, 'sthant1': 323, 'sthwoo1': 324,
    'strcuc1': 325, 'strfly1': 326, 'strsal1': 327, 'stvhum2': 328, 'subfly': 329,
    'sumtan': 330, 'swaspa': 331, 'swathr': 332, 'tenwar': 333, 'thbeup1': 334,
    'thbkin': 335, 'thswar1': 336, 'towsol': 337, 'treswa': 338, 'trogna1': 339,
    'trokin': 340, 'tromoc': 341, 'tropar': 342, 'tropew1': 343, 'tuftit': 344,
    'tunswa': 345, 'veery': 346, 'verdin': 347, 'vigswa': 348, 'warvir': 349,
    'wbwwre1': 350, 'webwoo1': 351, 'wegspa1': 352, 'wesant1': 353, 'wesblu': 354,
    'weskin': 355, 'wesmea': 356, 'westan': 357, 'wewpew': 358, 'whbman1': 359,
    'whbnut': 360, 'whcpar': 361, 'whcsee1': 362, 'whcspa': 363, 'whevir': 364,
    'whfpar1': 365, 'whimbr': 366, 'whiwre1': 367, 'whtdov': 368, 'whtspa': 369,
    'whwbec1': 370, 'whwdov': 371, 'wilfly': 372, 'willet1': 373, 'wilsni1': 374,
    'wiltur': 375, 'wlswar': 376, 'wooduc': 377, 'woothr': 378, 'wrenti': 379,
    'y00475': 380, 'yebcha': 381, 'yebela1': 382, 'yebfly': 383, 'yebori1': 384,
    'yebsap': 385, 'yebsee1': 386, 'yefgra1': 387, 'yegvir': 388, 'yehbla': 389,
    'yehcar1': 390, 'yelgro': 391, 'yelwar': 392, 'yeofly1': 393, 'yerwar': 394,
    'yeteup1': 395, 'yetvir': 396,
}
if USE_NOCALL:
    BIRD_CODE.update({'nocall': 397})

NUM_TARGETS = len(BIRD_CODE)
TARGET_COLS = list(BIRD_CODE.keys())
TARGET_COLS_WITHOUT_NOCALL = TARGET_COLS[:-1] if USE_NOCALL else TARGET_COLS[:]

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

root_dir = Path.cwd().parent
CSV_TRA_META = str(root_dir.joinpath(f'./input/birdclef-2021/train_metadata.csv'))
CSV_TRA_SOUNDSCAPE = str(root_dir.joinpath(f'./input/birdclef-2021/train_soundscape_labels.csv'))
CSV_SAMPLE_SUB = str(root_dir.joinpath(f'./input/birdclef-2021/sample_submission.csv'))
CSV_TEST = str(root_dir.joinpath(f'./input/birdclef-2021/test.csv'))

DIR_TRA_SOUNDSCAPES = str(root_dir.joinpath(f'./input/birdclef-2021/train_soundscapes/'))
DIR_TRA_SHORT_AUDIO = str(root_dir.joinpath(f'./input/birdclef-2021/train_short_audio/'))
DIR_TEST_SOUNDSCAPES = str(root_dir.joinpath(f'./input/birdclef-2021/test_soundscapes/'))
DIR_MEL = str(root_dir.joinpath(f'./working/mel_spectrogram/'))
# DIR_MEL = str(root_dir.joinpath(f'./working/mel_spec_fmin500/'))
DIR_WORKING = str(root_dir.joinpath(f'./working/'))
DIR_OUTPUT = str(root_dir.joinpath(f'./working/output/'))
DIR_VISUALIZE = str(root_dir.joinpath(f'./working/visualize/'))
DIR_BG_NOISE = str(root_dir.joinpath(f'./working/my_bg_noise/'))
DIR_ENERGY_PROBS = str(root_dir.joinpath(f'./working/energy_probs/'))
DIR_MANUAL_FILES = str(root_dir.joinpath(f'./working/manual_files/'))
DIR_MEL_BACKUP = str(root_dir.joinpath(f'./working/mel_backup/'))
DIR_FREE_FIELD1010 = str(root_dir.joinpath(f'./input/BG_noise_dataset/freefield1010/'))
DIR_TRAINED_MODELS = str(root_dir.joinpath(f'./input/chbird-kfolds/'))

CSV_TRA_MEL_META = str(root_dir.joinpath(f"./working/df_tra_mel_meta.csv"))
CSV_PSEUDO = str(root_dir.joinpath(f"./working/df_pseudo.csv"))

# SILENT_OGG means completely silent, nothing seen in spectrogram
SILENT_OGG = ['XC208883.ogg', 'XC579430.ogg', 'XC282102.ogg', 'XC580627.ogg', 'XC439459.ogg', 'XC141187.ogg',
              'XC590621.ogg', 'XC141034.ogg', 'XC154485.ogg']
# BAD_OGG means hardly see useful info from spectrogram
"""
BAD_OGG = ['XC208852.ogg', 'XC153315.ogg', 'XC153278.ogg', 'XC544809.ogg', 'XC186856.ogg', 'XC181332.ogg',
           'XC182574.ogg', 'XC182575.ogg', 'XC153177.ogg']
BAD_OGG = ['XC153315.ogg', 'XC153278.ogg', 'XC181332.ogg',
           'XC182574.ogg', 'XC182575.ogg', 'XC153177.ogg']
"""
BAD_OGG = ['XC153315.ogg', 'XC153278.ogg',
           'XC182574.ogg', 'XC182575.ogg', 'XC153177.ogg']

# DIFFICULT_OGG means useful info inside, but too weak/noisy, or difficult to energy trim on useful segment
"""
DIFFICULT_OGG = ['XC372211.ogg', 'XC313916.ogg', 'XC291935.ogg', 'XC244402.ogg', 'XC215880.ogg', 'XC215883.ogg',
                 'XC215885.ogg', 'XC215917.ogg', 'XC215915.ogg', 'XC331634.ogg', 'XC152348.ogg', 'XC152766.ogg',
                 'XC343332.ogg', 'XC403247.ogg', 'XC473710.ogg', 'XC252702.ogg']
DIFFICULT_OGG = ['XC244402.ogg']
"""
DIFFICULT_OGG = ['XC244402.ogg', 'XC473710.ogg']

ABANDONED_OGG = SILENT_OGG + BAD_OGG + DIFFICULT_OGG

# Have checked silent segment with those files
SILENCE_CHECKED_FILES = ['XC364138.ogg', 'XC511289.ogg', 'XC516415.ogg', 'XC422943.ogg', 'XC301149.ogg',
                         'XC432599.ogg', 'XC598443.ogg', 'XC512266.ogg', 'XC575935.ogg', 'XC616347.ogg',
                         'XC535288.ogg', 'XC594146.ogg', 'XC598449.ogg', 'XC289375.ogg', 'XC535160.ogg',
                         'XC598468.ogg', 'XC405848.ogg', 'XC405849.ogg', 'XC140979.ogg', 'XC405850.ogg',
                         'XC596717.ogg', 'XC158213.ogg', 'XC509625.ogg', 'XC511877.ogg', 'XC359408.ogg',
                         'XC567462.ogg', 'XC535153.ogg', 'XC535156.ogg', 'XC535157.ogg', 'XC475095.ogg',
                         'XC511295.ogg', 'XC598441.ogg', 'XC163319.ogg', 'XC512271.ogg', 'XC511321.ogg',
                         'XC335537.ogg', 'XC509632.ogg', 'XC328919.ogg', 'XC406105.ogg', 'XC556721.ogg',
                         'XC546682.ogg', 'XC598471.ogg', 'XC594762.ogg', 'XC137610.ogg', 'XC153177.ogg',
                         'XC535073.ogg', 'XC535078.ogg', 'XC535079.ogg', 'XC535080.ogg', 'XC535082.ogg',
                         'XC535085.ogg', 'XC535088.ogg', 'XC535089.ogg', 'XC535090.ogg', 'XC535091.ogg',
                         'XC535655.ogg', 'XC154192.ogg', 'XC598462.ogg', 'XC511283.ogg', 'XC535195.ogg',
                         'XC320841.ogg', 'XC512250.ogg', 'XC511313.ogg', 'XC410553.ogg', 'XC255323.ogg',
                         'XC511279.ogg', 'XC169454.ogg', 'XC598458.ogg', 'XC313231.ogg', 'XC575333.ogg',
                         'XC598448.ogg', 'XC518213.ogg', 'XC512248.ogg', 'XC518214.ogg', 'XC535260.ogg',
                         'XC535261.ogg', 'XC457489.ogg', 'XC117799.ogg', 'XC358657.ogg', 'XC318881.ogg',
                         'XC517921.ogg', 'XC320002.ogg', 'XC604129.ogg', 'XC544393.ogg', 'XC596709.ogg',
                         'XC494102.ogg', 'XC149583.ogg', 'XC468375.ogg', 'XC145611.ogg', 'XC535151.ogg',
                         'XC512264.ogg', 'XC509612.ogg', 'XC498044.ogg', 'XC512247.ogg', 'XC321071.ogg',
                         'XC616516.ogg', 'XC535045.ogg', 'XC535046.ogg', 'XC535047.ogg', 'XC535048.ogg',
                         'XC512273.ogg', 'XC477349.ogg', 'XC512275.ogg', 'XC512270.ogg', 'XC509626.ogg',
                         'XC562963.ogg', 'XC297670.ogg', 'XC534899.ogg', 'XC497500.ogg', 'XC499569.ogg',
                         'XC598466.ogg', 'XC511318.ogg', 'XC464836.ogg', 'XC538369.ogg', 'XC555116.ogg',
                         'XC137588.ogg', 'XC407022.ogg', 'XC511301.ogg', 'XC535113.ogg', 'XC509624.ogg',
                         'XC511281.ogg', 'XC511307.ogg', 'XC340366.ogg', 'XC469184.ogg', 'XC531820.ogg',
                         'XC308452.ogg', 'XC469985.ogg', 'XC584512.ogg', 'XC146291.ogg', 'XC287083.ogg',
                         'XC340534.ogg', 'XC361494.ogg', 'XC401087.ogg', 'XC509631.ogg', 'XC531579.ogg',
                         'XC232971.ogg', 'XC234341.ogg', 'XC511498.ogg', 'XC165209.ogg', 'XC598470.ogg',
                         'XC433434.ogg', 'XC431933.ogg', 'XC511310.ogg', 'XC511309.ogg', 'XC616939.ogg',
                         'XC574354.ogg', 'XC509629.ogg', 'XC511296.ogg', 'XC535303.ogg', 'XC535304.ogg',
                         'XC596708.ogg', 'XC535305.ogg', 'XC535306.ogg', 'XC535307.ogg', 'XC535310.ogg',
                         'XC598455.ogg', 'XC598461.ogg', 'XC560071.ogg', 'XC518083.ogg', 'XC518258.ogg',
                         'XC509630.ogg', 'XC544535.ogg', 'XC323428.ogg', 'XC535141.ogg', 'XC598456.ogg',
                         'XC112866.ogg', 'XC406951.ogg', 'XC509627.ogg', 'XC584986.ogg', 'XC518211.ogg',
                         'XC511280.ogg', 'XC598460.ogg', 'XC117823.ogg', 'XC117678.ogg', 'XC198512.ogg',
                         'XC450520.ogg', 'XC487427.ogg', 'XC367439.ogg', 'XC137599.ogg', 'XC307532.ogg',
                         'XC377774.ogg', 'XC598463.ogg', 'XC598459.ogg', 'XC607456.ogg', 'XC149593.ogg',
                         'XC356012.ogg', 'XC549227.ogg', 'XC328921.ogg', 'XC596704.ogg', 'XC596722.ogg',
                         'XC564800.ogg', 'XC372621.ogg', 'XC511291.ogg', 'XC244539.ogg', 'XC509628.ogg',
                         'XC596710.ogg', 'XC351199.ogg', 'XC444146.ogg', 'XC509610.ogg', 'XC559273.ogg',
                         'XC431935.ogg', 'XC342741.ogg', 'XC511288.ogg', 'XC596707.ogg', 'XC509615.ogg']

MANUALED_OGG = []
"""
['XC112866.ogg', 'XC117678.ogg', 'XC117799.ogg', 'XC117823.ogg', 'XC137588.ogg', 'XC137599.ogg',
 'XC137610.ogg', 'XC140979.ogg', 'XC145611.ogg', 'XC146291.ogg', 'XC149583.ogg', 'XC149593.ogg',
 'XC154192.ogg', 'XC158213.ogg', 'XC163319.ogg', 'XC165209.ogg', 'XC169454.ogg', 'XC198512.ogg',
 'XC232971.ogg', 'XC234341.ogg', 'XC255323.ogg', 'XC287083.ogg', 'XC297670.ogg', 'XC301149.ogg',
 'XC307532.ogg', 'XC308452.ogg', 'XC313231.ogg', 'XC318881.ogg', 'XC320002.ogg', 'XC320841.ogg',
 'XC321071.ogg', 'XC323428.ogg', 'XC328919.ogg', 'XC328921.ogg', 'XC335537.ogg', 'XC340366.ogg',
 'XC340534.ogg', 'XC342741.ogg', 'XC351199.ogg', 'XC356012.ogg', 'XC358657.ogg', 'XC359408.ogg',
 'XC361494.ogg', 'XC364138.ogg', 'XC367439.ogg', 'XC372621.ogg', 'XC377774.ogg', 'XC405848.ogg',
 'XC405849.ogg', 'XC405850.ogg', 'XC406105.ogg', 'XC406951.ogg', 'XC407022.ogg', 'XC410553.ogg',
 'XC422943.ogg', 'XC431933.ogg', 'XC431935.ogg', 'XC432599.ogg', 'XC433434.ogg', 'XC444146.ogg',
 'XC450520.ogg', 'XC457489.ogg', 'XC464836.ogg', 'XC468375.ogg', 'XC469184.ogg', 'XC469985.ogg',
 'XC475095.ogg', 'XC477349.ogg', 'XC487427.ogg', 'XC494102.ogg', 'XC497500.ogg', 'XC498044.ogg',
 'XC499569.ogg', 'XC509610.ogg', 'XC509612.ogg', 'XC509615.ogg', 'XC509624.ogg', 'XC509625.ogg',
 'XC509626.ogg', 'XC509627.ogg', 'XC509628.ogg', 'XC509629.ogg', 'XC509630.ogg', 'XC509631.ogg',
 'XC509632.ogg', 'XC511279.ogg', 'XC511280.ogg', 'XC511281.ogg', 'XC511283.ogg', 'XC511288.ogg',
 'XC511289.ogg', 'XC511291.ogg', 'XC511295.ogg', 'XC511296.ogg', 'XC511301.ogg', 'XC511307.ogg',
 'XC511309.ogg', 'XC511310.ogg', 'XC511313.ogg', 'XC511318.ogg', 'XC511321.ogg', 'XC511498.ogg',
 'XC511877.ogg', 'XC512247.ogg', 'XC512248.ogg', 'XC512250.ogg', 'XC512264.ogg', 'XC512266.ogg',
 'XC512270.ogg', 'XC512271.ogg', 'XC512273.ogg', 'XC512275.ogg', 'XC516415.ogg', 'XC517921.ogg',
 'XC518083.ogg', 'XC518211.ogg', 'XC518213.ogg', 'XC518214.ogg', 'XC518258.ogg', 'XC531579.ogg',
 'XC531820.ogg', 'XC534899.ogg', 'XC535045.ogg', 'XC535046.ogg', 'XC535047.ogg', 'XC535048.ogg',
 'XC535073.ogg', 'XC535078.ogg', 'XC535079.ogg', 'XC535080.ogg', 'XC535082.ogg', 'XC535085.ogg',
 'XC535088.ogg', 'XC535113.ogg', 'XC535141.ogg', 'XC535160.ogg', 'XC535195.ogg', 'XC535260.ogg',
 'XC535261.ogg', 'XC535288.ogg', 'XC535303.ogg', 'XC535304.ogg', 'XC535305.ogg', 'XC535306.ogg',
 'XC535307.ogg', 'XC535655.ogg', 'XC538369.ogg', 'XC544393.ogg', 'XC544535.ogg', 'XC546682.ogg',
 'XC549227.ogg', 'XC556721.ogg', 'XC559273.ogg', 'XC560071.ogg', 'XC562963.ogg', 'XC564800.ogg',
 'XC567462.ogg', 'XC575333.ogg', 'XC584512.ogg', 'XC584986.ogg', 'XC594146.ogg', 'XC594762.ogg',
 'XC596704.ogg', 'XC596707.ogg', 'XC596708.ogg', 'XC596709.ogg', 'XC596710.ogg', 'XC596717.ogg',
 'XC596722.ogg', 'XC598441.ogg', 'XC598443.ogg', 'XC598448.ogg', 'XC598449.ogg', 'XC598455.ogg',
 'XC598456.ogg', 'XC598458.ogg', 'XC598459.ogg', 'XC598460.ogg', 'XC598461.ogg', 'XC598462.ogg',
 'XC598463.ogg', 'XC598466.ogg', 'XC598468.ogg', 'XC598470.ogg', 'XC598471.ogg', 'XC604129.ogg',
 'XC616516.ogg', 'XC616939.ogg']
"""

