from analysis.event_analysis_by_question import *

def main():
    QEA = QuestionEventAnalyzer()
    '''q_id1 = 'question_0d1f3326-f81e-49e5-b796-7e47151fe3f8'
    q_id2 = 'question_90b19eaf-a1b3-4f82-9e7e-fa67a570190f'
    QEA.preprocess_event_data(filter_only_first_attempt=True, filter_by_min_n_answers=2,
                              filter_by_min_n_students=2)
    questions_list = QEA.processed_questions_list
    QEA.get_common_mistakes_df(questions_list,LOs='all')'''


    #QEA.get_question_features(q_id1,is_plot_mistake_description=True, save_name=q_id+'_hist.png')
    load_results=True
    plot_grid_heat_map=True
    '''EA=EventsAnalyzer()
    EA.load_data()
    EA.get_all_questions_event_general_description(min_num_students=150, min_answers=3, max_answers=50, is_plot=True)
    print(EA.top_questions)'''

    #arrange questions by LOs:
    #lo_index = df.from_csv(os.path.join(DATA_ROOT, 'LOs_order_index_fraction.csv'))
    #\todo - CONTINUE HERE -  (1) arrange questions according to LO and question index (2) per LO- run all questions against each other (hitmap) and pairwise.
    SEA = SystematicEventAnalyzer()

    SEA.preprocess_event_data(load_processed_data=True,
                              processed_data_file_name='events_table_math_processed_first_attempt.csv',
                              filter_only_first_attempt=True,
                              filter_by_min_n_answers=False,
                              filter_by_min_n_students=1,
                              add_error_type_to_data=False)
    questions_list = SEA.processed_questions_list
            #pd.MultiIndex.from_arrays([res_index.LO_subject_index,res_index.question_index])'''
    #questions_list=questions_list[:20]
    results_items = ['all']#'['mi', 'chi2', 'p', 'dof', 'n_students']
    results_items = ['chi2_fixed','mi_fixed', 'n_joint_responses',	'n_students_per_joint_responses',	'max_n_students_joint_responses']
    if load_results:
        full_results=df.from_csv(os.path.join(OUTPUT_DIR, 'all_vs_all_chi2_results_full.csv'), index_col=['q1','q2'])
        res={}

        for item in results_items:
            item_results=full_results[item].unstack()
            #item_results.index=pd.MultiIndex.from_arrays([full_results])
            res[item]=item_results

            #res[item] = df.from_csv(os.path.join(OUTPUT_DIR, 'all_vs_all_chi2_results_%s.csv' % item))
    else:

        res, full_results =SEA.all_vs_all_comparison(questions_list, is_save_to_csv=True)

    for item in results_items:
        results=res[item]
        res_index = SEA.question_indexes.loc[results.index]
        res_index.set_index('question_index',inplace=True)


        res_columns = SEA.question_indexes.loc[results.columns]
        results.index = res_index.question_index
        results.sort_index(inplace=True)
        results.columns = res_index.question_index
        results.sort_index(axis=1, inplace=True)

        results.dropna(how='all',inplace=True)
        axis1 = res_index.LO_subsubject_index.loc[results.index]
        axis2 = res_index.LO_subject_index.loc[results.index]
        if plot_grid_heat_map:
            grid_heatmap(results, index1=axis1, index2=axis2, title=item, save_name='all_vs_all_chi2_results_%s.png' % item)



    #questions_list = [q_id1, q_id2]
    pairwise_res = SEA.all_vs_all_comparison(questions_list, is_save_to_csv=True)
    pairwise_res=SEA.running_pairwise_question_comparison(questions_list, is_save_to_csv=True)
    QEA=QuestionEventAnalyzer()
    QEA.agg_by_question_attempts()

    #QEA.get_all_questions_features(is_plot_by_columns=True, is_save_csv=True)

    question_ind='question_39e75740-5f1f-4347-8870-44623b9b8e07'
#'question_629275a1-9403-48f9-853d-e5c981eecce8'#''question_4c3f85a1-0470-4fbb-9183-4ea4b4e53b57'#'question_25504e71-8563-4600-aa5e-be589b5e44c5'
    QEA.get_question_event_description(question_ind, is_plot=False)
    QEA.get_question_features(question_ind,is_plot_mistake_description=True)

main()