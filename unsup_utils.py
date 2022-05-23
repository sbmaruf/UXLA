import scipy
import numpy as np
from sklearn import mixture
from vis import plot_analysis

def select_mean_id(vals):
    mn = vals[0][0]
    _id = 0
    for idx, val in enumerate(vals):
        if val[0] < mn:
            mn = val[0] 
            _id = idx
    return _id

def unsupervised_sample_selection(
                    losses, 
                    n_mixture_component,
                    n_mixture_select,
                    threshold, 
                    covariance_type,
                    noise_model=None
    ):
    losses = np.array(losses)
    losses = losses.reshape(-1,1) if len(losses.shape) == 1 else losses
    if noise_model is None:
        noise_model = mixture.GaussianMixture(
                        n_components = n_mixture_component, 
                        covariance_type = covariance_type, 
                        tol = 1e-5,
                        reg_covar = 1e-5,
                        random_state = 1234
                    )
        noise_model.fit(losses)
    prob = noise_model.predict_proba(losses) 
    ret_index_list = None
    for i in range(n_mixture_select):
        min_class_index = noise_model.means_.argmin()
        prob1 = prob[:,min_class_index]
        pred = prob1 > threshold[i]
        index_list = list(range(len(losses)))
        np_index_list = np.array(index_list)
        np_index_list = np_index_list[pred]
        ret_index_list = np_index_list if ret_index_list is None else np.append(ret_index_list, np_index_list, axis=0)
        noise_model.means_[min_class_index, 0] = 1000000000
    ret_index_list = np.array(list(set(list(ret_index_list))))
    assert len(set(list(ret_index_list))) == len(list(ret_index_list)) 
    return ret_index_list, noise_model


def select_samples_with_GMM(
        args,
        dict_key,
        loss_dict, 
        path, 
        bin_increment, 
        noise_threshold=0, 
        min_length_restriction=10, 
        max_length_restriction=150,
        mode="train",
        logger=None,
        debug=0
    ):
    
    logger.info("Sample selection from :: {}".format(dict_key))
    values = loss_dict[dict_key]

    all_lengths = []
    loss_list = []
    
    for val in values:
        loss_list.append((val[0]))
        all_lengths.append(val[1])

    tota_number_of_samples = len(loss_list)
    logger.info("Total number of sample : {}".format(tota_number_of_samples))
    indexes, noise_model = unsupervised_sample_selection(
                    loss_list, 
                    args.n_mixture_component,
                    args.n_mixture_select,
                    args.posterior_threshold, 
                    args.covariance_type
                )
    tota_number_of_selected_samples = len(indexes)
    logger.info("Total number of sample selected : {}".format(tota_number_of_selected_samples))

    if debug:
        correct_sample_cnt = 0
        noisy_sample_cnt = 0
        good_samples_cnt = 0
        cnt = 0
        selected_sentence_lengths = []
        for i in indexes:
            val = values[i]
            if val[2] > noise_threshold:
                noisy_sample_cnt += 1
            else:
                correct_sample_cnt += 1
            cnt += 1
            selected_sentence_lengths.append(val[1])
            if val[1] >= min_length_restriction and val[1] <= max_length_restriction:
                good_samples_cnt += 1
        if tota_number_of_selected_samples != 0:
            logger.info("Total Number of Noisy Samples : {} ({}%)".format(noisy_sample_cnt, round(noisy_sample_cnt/tota_number_of_selected_samples*100,2)))
            logger.info("Total Number of Correct Samples : {} ({}%)".format(correct_sample_cnt, round(correct_sample_cnt/tota_number_of_selected_samples*100,2)))
            logger.info("Total Number of Correct Sample in dataset : {} ({}%)".format(correct_sample_cnt, round(correct_sample_cnt/len(loss_list)*100,2)))
            logger.info("Good Samples Coverage : {} ({}%)".format(good_samples_cnt, round(good_samples_cnt/len(loss_list)*100,2)))
            
            sample_selection_acc = round(correct_sample_cnt/tota_number_of_selected_samples*100, 2)

            plot_analysis(
                dict_key = dict_key, 
                loss_dict = loss_dict, 
                all_lengths = all_lengths,
                selected_sentence_lengths = selected_sentence_lengths,
                path = path, 
                bin_increment = bin_increment, 
                noise_threshold = noise_threshold,
                mode = mode,
                min_length_restriction = min_length_restriction, 
                max_length_restriction = max_length_restriction,
                sample_selection_acc = sample_selection_acc
            )
    
    return indexes, noise_model




def select_data_from_logit(
        args,
        dict_key,
        logit_dict, 
        loss_dict,
        path, 
        bin_increment, 
        top_k,
        noise_threshold=0, 
        min_length_restriction=10, 
        max_length_restriction=150,
        mode="train",
        logger=None,
        debug=0,
        isGMM=0
    ):
    
    logger.info("Sample selection from :: {}".format(dict_key))
    sentence_logits = logit_dict[dict_key]
    
    values = loss_dict[dict_key]
    all_lengths = []
    loss_list = []
    
    for val in values:
        # if val[1] > max_length_restriction:
        #     continue
        loss_list.append((val[0]))
        all_lengths.append(val[1])

        
    logit_avg_idx_list = []
    for idx, logits in enumerate(sentence_logits):
        __sum = 0
        for logit in logits:
            logit_softmax = scipy.special.softmax(logit, axis=-1)
            __sum += logit_softmax.max()
        __avg = __sum / len(logits)
        logit_avg_idx_list.append((__avg, idx))
    logit_avg_idx_list = sorted(logit_avg_idx_list, key=lambda logit_sum_idx_list: logit_sum_idx_list[0], reverse=True)
    total_selected_sample = (len(logit_avg_idx_list)*top_k)//100

    logger.info("Total number of sample selected (by max sorting) : {}".format(total_selected_sample))
    
    if isGMM == 1:
        indexes = []
        confidence = []
        for i in range(total_selected_sample):
            # indexes.append(logit_avg_idx_list[i][1])
            confidence.append([ -logit_avg_idx_list[i][0] ])
        confidence = np.array(confidence)
        I, noise_model = unsupervised_sample_selection(
                        confidence, 
                        args.n_mixture_component,
                        args.n_mixture_select,
                        args.posterior_threshold, 
                        args.covariance_type
                    )
        indexes = []
        for idx in I:
            indexes.append(logit_avg_idx_list[idx][1])
            
    else:
        indexes = []
        for i in range(total_selected_sample):
            indexes.append(logit_avg_idx_list[i][1])


    ######################################
    if debug:
        tota_number_of_selected_samples = len(indexes)
        logger.info("Total number of sample selected (if GMM applied) : {}".format(tota_number_of_selected_samples))
        correct_sample_cnt = 0
        noisy_sample_cnt = 0
        good_samples_cnt = 0
        cnt = 0
        selected_sentence_lengths = []
        for i in indexes:
            val = values[i]
            if val[2] > noise_threshold:
                noisy_sample_cnt += 1
            else:
                correct_sample_cnt += 1
            cnt += 1
            selected_sentence_lengths.append(val[1])
            if val[1] >= min_length_restriction and val[1] <= max_length_restriction:
                good_samples_cnt += 1

        if tota_number_of_selected_samples != 0:
            logger.info("Total Number of Noisy Samples : {} ({}%)".format(noisy_sample_cnt, round(noisy_sample_cnt/tota_number_of_selected_samples*100,2)))
            logger.info("Total Number of Correct Samples : {} ({}%)".format(correct_sample_cnt, round(correct_sample_cnt/tota_number_of_selected_samples*100,2)))
            logger.info("Total Number of Correct Sample in dataset : {} ({}%)".format(correct_sample_cnt, round(correct_sample_cnt/len(loss_dict)*100,2)))
            
            sample_selection_acc = round(correct_sample_cnt/tota_number_of_selected_samples*100, 2)
            plot_analysis(
                dict_key = dict_key, 
                loss_dict = loss_dict, 
                all_lengths = all_lengths,
                selected_sentence_lengths = selected_sentence_lengths,
                path = path, 
                bin_increment = bin_increment, 
                noise_threshold = noise_threshold,
                mode = mode,
                min_length_restriction = min_length_restriction, 
                max_length_restriction = max_length_restriction,
                sample_selection_acc = sample_selection_acc
            )
    
    return indexes