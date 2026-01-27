import ijson
with open('_annotations_COCO_final.json', 'rb') as f:
    for ann in ijson.items(f, 'annotations.item'):
        if ann['image_id'] == 100000:
            print('Caption:', ann.get('caption', ''))
            print('Bbox:', ann.get('bbox'))
            seg = ann.get('segmentation')
            print('Seg type:', type(seg))
            if seg:
                print('Seg len:', len(seg))
                if len(seg) > 0:
                    print('Seg[0] type:', type(seg[0]))
                    print('Seg[0][:10]:', seg[0][:10] if len(seg[0]) > 10 else seg[0])
            kp = ann.get('keypoints')
            print('KP type:', type(kp))
            if kp:
                print('KP len:', len(kp))
            break
