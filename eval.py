import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
import os

ckpt_save = 'model_108_loss_0.0XXXX.pth'

method = 'MGCNet'



for _data_name in ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']:
    print("eval-dataset:", _data_name)

    mask_root = os.path.join(
        r'',
        _data_name,
        'GT'
    )

    pred_root = os.path.join(
        r'',
        _data_name
    )

    print("Mask Path:", mask_root)
    print("Pred Path:", pred_root)

    pred_name_list = sorted(os.listdir(pred_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for pred_name in tqdm(pred_name_list, total=len(pred_name_list)):
        mask_path = os.path.join(mask_root, pred_name)
        pred_path = os.path.join(pred_root, pred_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
        "MAE": mae,
    }

    print(results)
    file = open(r"\eval_results.txt", "a")
    file.write(method+' '+_data_name+' '+str(ckpt_save)+'\n')

print("Eval finished!")