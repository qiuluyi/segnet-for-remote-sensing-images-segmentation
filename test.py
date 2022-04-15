import torch.optim.lr_scheduler
import torch.nn.init

from camnet import SegNet


from preprocess import *
from PIL import Image

try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

"""
def retest(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    DATA_FOLDER Predict_FOLDER NSDM_FOLDER np.concatenate(io.imread(DATA_FOLDER.format(id)),io.imread(Predict_FOLDER.format(id)), axis=2)
    # test_images = (1 / 255 * np.asarray(np.concatenate((io.imread(Predict_FOLDER.format(id)), io.imread(NSDM_FOLDER.format(id)),io.imread(DATA_FOLDER.format(id))), axis=2), dtype='float32') for id in test_ids)
    test_images = (1 / 255 * np.asarray(np.concatenate((io.imread(DATA_FOLDER.format(id)),io.imread(Predict_FOLDER.format(id))), axis=2), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    num=0
    #eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    list0 = []
    # Switch the network to inference mode
    net.eval()

    for img, gt in tqdm(zip(test_images, test_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))
        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):
            # Display in progress results
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                _pred = np.argmax(pred, axis=-1)
                #fig = plt.figure()
                #fig.add_subplot(1, 3, 1)
                #plt.imshow(np.asarray(255 * img, dtype='uint8'))
                #fig.add_subplot(1, 3, 2)
                #plt.imshow(convert_to_color(_pred))
                #fig.add_subplot(1, 3, 3)
                #plt.imshow(gt)
                #clear_output()
                #plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            # Do the inference
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)
        pre_img=convert_to_color(pred)
        im = Image.fromarray(pre_img)
        print("test_id",test_ids[num])
        im.save(Pred_FOLDER.format(test_ids[num]))
        num=num+1
        # Display the result
        # clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(convert_to_color(pred))
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(gt)
        # plt.show()

        all_preds.append(pred)
        all_gts.append(gt)

        # clear_output()
        # Compute some metrics
        # metrics(pred.ravel(), gt_e.ravel())
    meanF1score=0
    # accuracy, meanF1score = metrics(np.concatenate([p.ravel() for p in all_preds]),
    #                                            np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, meanF1score,all_preds, all_gts
    else:
        return accuracy,meanF1score


"""

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    #PRE_FOLDER = './checkpoint12/epoch60/top_mosaic_09cm_area{}.tif'
    num=0
    list0=[]
    meanF1score=0
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:1")
    net.to(device)
    # Switch the network to inference mode
    net.eval()
    # (2767,2428,3)
    for img, gt in tqdm(zip(test_images, test_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):
            # Display in progress results
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                _pred = np.argmax(pred, axis=-1)

                #fig = plt.figure()
                #fig.add_subplot(1, 3, 1)
                #plt.imshow(np.asarray(255 * img, dtype='uint8'))
                #fig.add_subplot(1, 3, 2)
                #plt.imshow(convert_to_color(_pred))
                #fig.add_subplot(1, 3, 3)
                #plt.imshow(gt)
                #clear_output()
                #plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            # Do the inference
            net.to(device)
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)

        # 写入文件夹
        pre_img=convert_to_color(pred)
        im = Image.fromarray(pre_img)
        print("test_id",test_ids[num])
        im.save(Pred_FOLDER.format(test_ids[num]))
        num=num+1
        # Display the result
        # clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(convert_to_color(pred))
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(gt)
        # plt.show()

        all_preds.append(pred)
        all_gts.append(gt)

        # metrics(pred.ravel(), gt_e.ravel())

    # accuracy, meanF1score,list0 = metrics(np.concatenate([p.ravel() for p in all_preds]),
    #                            np.concatenate([p.ravel() for p in all_gts]).ravel())
    # print("accuracy",accuracy)
    # print("meanF1score",meanF1score)

    if all:
        return accuracy,meanF1score, all_preds, all_gts
    else:
        return accuracy,meanF1score

def test_method():
    # net = SegNet()
    # net = DUNet()
    net = SegNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load('./checkpoint22/two_stage_epoch15')
    net.load_state_dict(checkpoint)

    test_ids = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
    print("Tiles for testing : ", test_ids)
    # acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
    # acc, meanF1score = fustest(net, test_ids, all=False, stride=min(WINDOW_SIZE))
    # print(acc)
    # print(meanF1score)
    test(net, test_ids)




def convert(test_ids):
    palette = {0: (255, 255, 255),  # Impervious surfaces (white)
               1: (0, 0, 255),  # Buildings (blue)
               2: (0, 255, 255),  # Low vegetation (cyan)
               3: (0, 255, 0),  # Trees (green)
               4: (255, 255, 0),  # Cars (yellow)
               5: (255, 0, 0),  # Clutter (red)
               6: (0, 0, 0)}  # Undefined (black)
    invert_palette = {v: k for k, v in palette.items()}
    for id in test_ids:
        pred = sitk.ReadImage(Predict_FOLDER.format(id))
        pred = sitk.GetArrayFromImage(pred)
        # pred = np.transpose(pred, (1, 2, 0))
        pred=pred[0]
        pred = convert_to_color(pred, palette=palette)
        # im = np.asarray(255 * pred, dtype='float32')
        im = Image.fromarray(pred)
        im.save(Predict_FOLDER.format(id))

if __name__=="__main__":
    train_ids = ['1', '3', '5', '7', '11', '13', '15', '17', '21', '23', '26', '28', '30', '32', '34', '37']

    # train_ids = ['1', '6', '7', '8', '11', '12', '13', '15', '17', '20', '21', '22', '28', '30', '32', '33']
    test_ids = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
    # test_ids = ['1', '6', '7', '8', '11', '12', '13', '15', '17', '20', '21', '22', '28', '30', '32', '33', '34']
    # test_set = ISPRS_dataset(test_ids, cache=CACHE)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)
    # print(len(all_id))
    # convert(all_id)
    test_method()
