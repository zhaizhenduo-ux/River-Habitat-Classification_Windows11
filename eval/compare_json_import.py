import json
import numpy as np
import pandas as pd


def IoU(true_box, pred_box):

	[xmin1, ymin1, xmax1, ymax1] = [true_box[0],true_box[1],true_box[0]+true_box[2],true_box[1]+true_box[3]]
	[xmin2, ymin2, xmax2, ymax2] = [int(pred_box[0]),int(pred_box[1]),int(pred_box[0]+pred_box[2]),int(pred_box[1]+pred_box[3])]
	area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
	area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
	xmin_inter = max(xmin1, xmin2)
	xmax_inter = min(xmax1, xmax2)
	ymin_inter = max(ymin1, ymin2)
	ymax_inter = min(ymax1, ymax2)
	if xmin_inter > xmax_inter or ymin_inter > ymax_inter:
	    return 0
	area_inter = (xmax_inter - xmin_inter) * (ymax_inter - ymin_inter)
	return float(area_inter) / (area1 + area2 - area_inter)

def calculate_precis_recall(true_bbox,pred_bbox,iou):
    fn = 0
    fp = 0
    tp = 0
    tp_cate = 0
    fp_cate = 0
    fn_cate = 0

    total_pred = len(pred_bbox)
    nneg = lambda x :max(0,x)
    # print(len(true_bbox))
    if (len(true_bbox)*len(pred_bbox)==0):
        fn = len(true_bbox)
        fp = len(pred_bbox)
        tp = 0
        print(fp)
    else:
        for t_bbox in true_bbox:
            iou_val = []
            positive = [] 
            for p_bbox in pred_bbox:
                iou_val.append(IoU(t_bbox,p_bbox))
                if IoU(t_bbox,p_bbox) > iou:
                	positive = p_bbox

            if sum(np.array(iou_val)>iou)==0:
                fn += 1
            else :
                tp+=1
                if t_bbox[-1] == positive[-1]:
                	if t_bbox[-1] == 1:
                		tp_cate +=1
                else:
                	fp_cate += 1
                taken = iou_val.index(max(iou_val))
                pred_bbox.remove(pred_bbox[taken])
    fp = total_pred-tp
    return tp,fp,fn,tp_cate,fp_cate,fn_cate

def compare(prediction_dir,ground_truth_dir,output_dir,results_dir,threshhold = 0.5,iou = 0):
	bird_prediction_num = 0
	pred_json = open(prediction_dir)
	prediction = json.load(pred_json)
	gt_json = open(ground_truth_dir)
	ground_truth = json.load(gt_json)

	prediction_num = 0

	log = open(output_dir,'w')
	log.write('F1 score of the model')
	image_name = ground_truth['images']
	gt_id = []
	ground_truth = ground_truth['annotations']

	image = []
	for img in image_name:
		image.append(img['file_name'])
		# gt_id.append(img['id'])
	image_list = []
	image_dic = {}
	pred_list = []
	cate_list = []
	gt_list = []
	empty_pred = 0
	gt_image_dic = {}

	# for id in gt_id:
	# 	gt_list.append([])

	for img in image_name:
		gt_image_dic[img['id']] = img['file_name']

	if prediction == []:
		pred_list = []

	else:
		for pred in prediction:
			if pred['score'] >threshhold:
				if pred['category_id'] == 1:
					bird_prediction_num += 1
				pred_bbox = pred['bbox']
				pred_img = pred['image_id']
				pred_cat = pred['category_id']
				pred_bbox.append(pred['score'])
				pred_bbox.append(pred_cat)
				prediction_num += 1
				if (pred_img in image_list):
					ind = (image_list.index(pred_img))
					pred_list[ind].append(np.asarray(pred_bbox).tolist())

				else:
					image_list.append(pred_img)
					tmp=[]
					tmp.append(np.asarray(pred_bbox).squeeze().tolist())
					pred_list.append(tmp)

	for gt in ground_truth:
		gt_bbox = gt['bbox']
		gt_img = gt['image_id']
		gt_bbox.append(gt['category_id'])
		if (gt_img in gt_id):
			ind = (gt_id.index(gt_img))
			gt_list[ind].append(np.asarray(gt_bbox).squeeze().tolist())
		else:
			gt_id.append(gt_img)
			tmp = []
			tmp.append(np.asarray(gt_bbox).tolist())
			gt_list.append(tmp)


	if (len(gt_id)!=len(image_list)):
		for idx,test in enumerate(gt_id):
			if (test in image_list):
				continue
			else:
				empty_pred+=len(gt_list[gt_id.index(test)])

				for img in image_name:
					if (img['id']==test):
						print ('The missing data are:'+ img['file_name'])
						log.write('\nThe missing data are:'+ img['file_name'])
	else:
		print ('Two data set equal')
	
	false_pred = []
	true_pred = []
	truth_neg =[]
	ture_cate_list= []
	precision_per_image =[]
	recall_per_image = []
	total_num = []
	image_names = []
	tp_cate_list = []
	fp_cate_list = []
	fn_cate_list = []
	import cv2


	for idx in image_list:
		if (idx in gt_id):
			gt_index = gt_id.index(idx)
			pred_index = image_list.index(idx)
			tp,fp,fn,tp_cate,fp_cate,fn_cate = calculate_precis_recall(gt_list[gt_index],pred_list[pred_index],iou)
			false_pred.append(fp)
			true_pred.append(tp)
			truth_neg.append(fn)
			tp_cate_list.append(tp_cate)
			fp_cate_list.append(fp_cate)
			fn_cate_list.append(fn_cate)

			total_num.append(tp+fn)
			precision_this_image = (1.0*tp)/(1.0*tp+1.0*fp)
			recall_this_image = (1.0*tp)/(1.0*tp+1.0*fn)
			recall_per_image.append(recall_this_image)
			precision_per_image.append(precision_this_image)
			image_names.append(gt_image_dic[idx])

	precision = (1.0*np.sum(true_pred))/((1.0*prediction_num)) 
	recall = (1.0*np.sum(true_pred)/(1.0*(np.sum(true_pred)+np.sum(truth_neg))))
	f1_score = 2*precision*recall/(precision+recall)
	cate_precision = (1.0*np.sum(tp_cate_list))/(1.0*bird_prediction_num)
	cate_recall  = (1.0*np.sum(tp_cate_list))/(1.0*(np.sum(true_pred)+np.sum(truth_neg)))
	cate_f1_score = 2*cate_precision*cate_recall/(cate_precision+cate_recall)
	cate_error = (1.0*np.sum(fp_cate_list))/(1.0*(np.sum(true_pred)+np.sum(truth_neg)))

	columns = ['image','birds','tp','fp','fn','precision','recall']
	dataframe = pd.DataFrame({'image':image_names,'birds':total_num,'tp':true_pred,'fp':false_pred,'fn':truth_neg,'precision':precision_per_image,'recall':recall_per_image})
	dataframe.to_csv(results_dir,index=False,sep=',',columns = columns)

	print ('The missing pred will be: '+str(empty_pred))
	print ('The precision will be'+str(precision))
	print ('The recall will be '+str(recall))
	print ('The f1 score will be '+str(2*precision*recall/(precision+recall)))
	print ('The cate_precision will be'+str(cate_precision))
	print ('The cate_recall will be '+str(cate_recall))
	print ('The cate_f1 score will be '+str(cate_f1_score))
	print ('The cate_error will be '+str(np.sum(fp_cate_list)))
	print(np.sum(true_pred),np.sum(false_pred))

	log.write('\nThe precision will be'+str(precision))
	log.write('\nThe recall will be '+str(recall))
	log.write('\nThe f1 score will be '+str(2*precision*recall/(precision+recall)))
	log.write('\nThe cate_precision will be'+str(cate_precision))
	log.write('\nThe cate_recall will be '+str(cate_recall))
	log.write('\nThe cate_f1 score will be '+str(cate_f1_score))
	log.write('\nThe cate_error will be '+str(np.sum(fp_cate_list)))
	log.close()
