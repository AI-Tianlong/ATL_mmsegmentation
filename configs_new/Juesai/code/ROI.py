def multi_ROI():
    label_2_path = '/opt/AI-Tianlong/openmmlab/mmsegmentation/configs_new/Juesai/shuimian1_mask'
    ROI_path = '/input_path/Roi/'

    label_list = find_data_list(label_2_path, suffix='.png')
    ROI_list = find_data_list(ROI_path, suffix='.png')

    for index in trange(len(label_list)):
        label_255 = np.array(Image.open(label_list[index]))

        ROI = np.array(Image.open(ROI_list[index]))

        final_label = label_255 * ROI
        final_label = Image.fromarray(final_label).save(
            os.path.join(label_2_path, os.path.basename(label_list[index])))


multi_ROI()
