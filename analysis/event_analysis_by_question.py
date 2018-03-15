from analysis.event_analysis import *
from scipy.stats import chi2_contingency

class QuestionEventAnalyzer(EventsAnalyzer):
    """
        analysis of CET raw events data (returned from .json by CetEventProcessing --> Event_Processing.CetEventProcessing.load())
        The analysis is by question, namely, for each question
         get_response_vector() calculates all responses and their frequencies
         get_question_features() calculates the ratio between common and non common responses to evaluate whether there are many common mistakes in students responses

        """
    def get_question_event_description(self,question_id, is_plot=False, is_clean_response=False):
        """ returns all responses of a specific question including whether the response was correct or not """
        if type(self.event_data)!= pd.core.frame.DataFrame:
            self.load_data()

        question_events_raw=self.event_data.loc[self.event_data['question_id']==question_id]

        responses_hist=df(question_events_raw['clean_response'].value_counts())

        is_correct_answer = [question_events_raw.loc[question_events_raw['clean_response']==r]['score'].sum()>0 for r in responses_hist.index]
        responses_hist.index = ['%s*' %r.split("'")[3] if is_correct_answer[i] else '%s' %r.split("'")[3] for  i,r in enumerate(responses_hist.index)]

        if is_plot:
            responses_hist.plot(kind='bar')
            plt.title(question_id)
            plt.savefig(os.path.join(OUTPUT_DIR, 'responses_hist_%s.png' %question_id))
        responses_hist['is_correct_answer'] = is_correct_answer

        return responses_hist

    def get_response_vector(self, question_id, first_attempt_only=True, non_common_response_threshold=0.02, max_number_of_answers=10, drop_correct_answer=True, clean_response=False):
        """
        This function filters all responses of a specific question and return the response vector of all students and their frequencty
        :param question_id:
        :param first_attempt_only:
        :param non_common_response_threshold:
        :param max_number_of_answers:
        :param drop_correct_answer:
        :param clean_response:
        :return:
        """
        mark_as_other=[]
        raw_responses = self.processed_event_data.loc[self.processed_event_data.question_id == question_id][
            ['student_id', 'clean_response', 'score']].copy()
        correct_answers = set(raw_responses.loc[raw_responses.score == 1]['clean_response'])
        raw_responses.set_index('student_id', drop=True, inplace=True)
        raw_responses = raw_responses['clean_response']
        raw_responses.name = question_id
        if drop_correct_answer:
            for r in correct_answers:
                raw_responses = raw_responses[raw_responses != r]

        #mark all non frequent responses as 'other' and sum over them
        raw_marginal_frequency = raw_responses.value_counts() / len(raw_responses)
        if len(raw_marginal_frequency)>max_number_of_answers:
            mark_as_other.extend(raw_marginal_frequency.index[max_number_of_answers:]) #leave only the 10 most common answers
        if len(raw_responses)==0:
            return df(), df()
        non_common_response_threshold=max(1/len(raw_responses), non_common_response_threshold) # if only one student answered mark as 'other'
        non_frequent_responses = raw_marginal_frequency.loc[raw_marginal_frequency < non_common_response_threshold].index
        mark_as_other.extend(non_frequent_responses)
        responses=raw_responses.copy()#.values=[r if r not in mark_as_other else 'other' for r in raw_responses.values ]
        for r in set(mark_as_other):
            responses[raw_responses==r]='other'



        #set maximum


        if not drop_correct_answer: #mark correct answers with a star
            marginal_frequency.index = ['%s*' %r if r in correct_answers else r for r in marginal_frequency.index]
            raw_responses=['%s*' %r if r in correct_answers else r for r in raw_responses]


        if clean_response:
            cleaner=lambda s: s[s.find(': ')+3:-2].replace("\\\\","").replace("frac","").replace("}{","/")
            raw_responses=raw_responses.apply(cleaner)

        if first_attempt_only:
            if len(raw_responses)>len(set(raw_responses.index)): #if there are duplicated indexes:
                raw_responses=raw_responses.reset_index().drop_duplicates(subset='student_id', keep='first').set_index('student_id')
                raw_responses=pd.Series(raw_responses)


        marginal_frequency = raw_responses.value_counts() / len(raw_responses)
        #marginal_frequency['other']=non_frequent_responses.sum()
        #marginal_frequency.drop(non_frequent_responses.index, inplace=True)
        return raw_responses, marginal_frequency

    def get_common_mistakes_df(self, questions_list, LOs='all', common_mistake_threshold=0.02, n_common_mistakes=5, save_name='temp_common_errors.csv', OVERRIDE=False):
        """ for each question in question list, calculates the most N common mistakes (n_common_mistakes),
        a common mistake is a (wrong) response given by at least t percent of the students (t=common_mistake_threshold)"""
        meta_data_columns = ['sQuestionPageID', 'sLOurl', 'LO_subject_index', 'LO_subsubject_index',
                             'num_of_questions_in_lo_session']

        md = self.meta_data[meta_data_columns].copy()
        md.index = self.meta_data.sElementID
        md = md.loc[questions_list]
        md.drop_duplicates(inplace=True)

        common_mistakes_df = df(index=pd.MultiIndex.from_product([questions_list,range(n_common_mistakes+1)]),columns=['mistake','p','n']+meta_data_columns)

        qLO_prev=0
        for question_id in questions_list:
            qLO=md.loc[question_id]['LO_subject_index']
            print('-%s - question_id' %qLO)
            if LOs!='all':
                if qLO in LOs:
                    pass
                else:
                    continue
            raw_responses, common_mistakes=self.get_response_vector(question_id, non_common_response_threshold=common_mistake_threshold, max_number_of_answers=n_common_mistakes,clean_response=True)
            '''responses_hist = self.get_question_event_description(question_id)
            correct_responses = responses_hist['clean_response'].loc[responses_hist['is_correct_answer'] == True]
            mistakes_count_hist = responses_hist['clean_response'].loc[responses_hist['is_correct_answer'] == False]

            mistakes_percent_hist = mistakes_count_hist / mistakes_count_hist.sum()

            common_mistakes = marginal_frequency.loc[mistakes_percent_hist>common_mistake_threshold]
            if len(common_mistakes)>n_common_mistakes:
                common_mistakes=common_mistakes.head(n_common_mistakes)'''
            for i,ind in enumerate(common_mistakes.index):
                common_mistakes_df.loc[question_id, i]['mistake']= ind
                common_mistakes_df.loc[question_id, i]['p'] = common_mistakes.loc[ind]
                common_mistakes_df.loc[question_id, i]['n'] = common_mistakes.loc[ind]*len(raw_responses)
                common_mistakes_df.loc[question_id, i][meta_data_columns] = md.loc[question_id]
        if qLO_prev!=qLO:
            common_mistakes_df.to_csv(os.path.join(OUTPUT_DIR, 'temp_common_mistakes_LO1-%s.csv' %qLO))
        qLO_prev = qLO
        #common_mistakes_df=common_mistakes_df.unstack()
        # add meta data

        common_mistakes.dropna(how='all',inplace=True)
        common_mistakes.reset_index(inplace=True)

        if OVERRIDE:
            common_mistakes_df.to_csv(os.path.join(OUTPUT_DIR,save_name))

        return common_mistakes_df

    def get_question_features(self, question_id, top_answers_threshold_list=[0.3,0.5,0.8], common_mistake_threshold=0.10, non_common_mistake_threshold=0.01, return_features='all', save_name='temp.png',is_plot_mistake_description=False ):
        """
        This function calculates the number of common and non_common mistakes in students responses to a specific CET question
        :param question_id:
        :param top_answers_threshold_list: for the percentile features
        :param common_mistake_threshold: min percent of students who made the same mistake to define the mistake as 'common'
        :param non_common_mistake_threshold: max percent of students who made the same mistake to define the mistake as 'non_common'
        :param return_features: which of the calculated features should be returned
        :param save_name:
        :param is_plot_mistake_description:
        :return:
        """
        F=dict() #question features dict
        responses_hist=self.get_question_event_description(question_id)
        correct_responses=responses_hist['clean_response'].loc[responses_hist['is_correct_answer']==True]
        mistakes_count_hist = responses_hist['clean_response'].loc[responses_hist['is_correct_answer'] == False]

        mistakes_percent_hist = mistakes_count_hist / mistakes_count_hist.sum()
        cumsum_mistakes_percent_hist = mistakes_percent_hist.cumsum()
        mistakes_desc=pd.concat([mistakes_count_hist,mistakes_percent_hist,cumsum_mistakes_percent_hist],axis=1)
        mistakes_desc.columns=['answer_count', 'answer_percent', 'percent_cumsum']

        F['n_wrong_answers']=len(mistakes_count_hist.index)
        F['n_correct_answers'] = len(correct_responses.index)
        F['percent_wrong_answers']=mistakes_count_hist.sum()/responses_hist['clean_response'].sum()
        F['mistakes_normalized_entropy'] = entropy(mistakes_percent_hist)/np.log(F['n_wrong_answers'])

        for t in top_answers_threshold_list:
            F['top%.02f' %t]=(cumsum_mistakes_percent_hist <= t).sum() + 1

        common_mistakes= mistakes_percent_hist.loc[mistakes_percent_hist >= common_mistake_threshold]
        non_common_mistakes= mistakes_percent_hist.loc[mistakes_percent_hist <= max(non_common_mistake_threshold, 1 / mistakes_count_hist.sum())]

        F['n_common_mistakes(>%.02f)' %common_mistake_threshold]=len(common_mistakes)
        F['percent_common_mistakes(>%.02f)' % common_mistake_threshold] = common_mistakes.sum()
        F['n_non_common_mistakes (<%.02f or 1 student)' %non_common_mistake_threshold] =len(non_common_mistakes)
        F['percent_non_common_mistakes (<%.02f or 1 student)' % non_common_mistake_threshold] = non_common_mistakes.sum()

        if is_plot_mistake_description:
            simple_df_plot(mistakes_desc,kind='bar',is_subplots=True,save_name='mistake_histogram_%s.png' %question_id,OVERRIDE=True, title=question_id)

        features=df.from_dict(F,orient='index')[0]
        features.name=question_id

        self.mistake_description=mistakes_desc
        if return_features=='all':
            self.question_event_features=features
            return features
        else:
            self.question_event_features = features[return_features]
            return features[return_features]

    def get_all_questions_features(self, questions=None, min_num_students=150, min_answers=5, max_answers=200 , is_save_csv=False):
        """
        This function explore students’ response to each question in order to detect common mistakes and possible mistake patterns
        :param questions:
        :param min_num_students:
        :param min_answers:
        :param max_answers:
        :param is_plot:
        :param is_plot_by_columns:
        :param plot_columns:
        :param is_save_csv:
        :return:
        """
        if not questions:
            self.load_data()
            top_questions = self.agg_by_question(min_num_students=min_num_students, min_answers=min_answers, max_answers=max_answers)
            questions=top_questions

        question_features = df(columns=questions)
        print('calculating featuers for top %i questions...' %len(questions))
        for question_id in questions:
            question_features[question_id]=self.get_question_features(question_id)
        question_features=question_features.T
        question_features['n_students']=self.question_description['students_per_question'].loc[questions]
        question_features.sort_values('n_students', ascending=False, inplace=True)
        question_features.sort_index(inplace=True)
        question_features['common_non_common_ratio'] = question_features['n_common_mistakes(>0.10)']/question_features['n_non_common_mistakes (<0.01 or 1 student)']
        self.question_features=question_features

        '''basic_question_features_list=['n_students','n_correct_answers','n_wrong_answers','percent_wrong_answers']
        percentile_question_features_list=['top0.30', 'top0.50', 'top0.80']
        entropy_question_features_list=['mistakes_normalized_entropy']
        advance_question_features_list=['n_common_mistakes(>0.10)','n_non_common_mistakes (<0.01 or 1 student)']
        advance_question_features_list = ['perent_common_mistakes(>0.10)', 'percent_non_common_mistakes (<0.01 or 1 student)']'''''

        if is_save_csv:
            self.question_features.to_csv(os.path.join(OUTPUT_DIR, 'Question_features.csv'))

        return question_features

    def plot_question_features(self,question_features=None, is_plot_by_columns=False, plot_columns='all',OVERRIDE=False):
        if 'question_features' in dir(self) and question_features is not None:
            question_features=self.question_features

        if not is_plot_by_columns:
            simple_df_plot(question_features, columns=plot_columns,args={'kind':'bar', 'reset_index':True},
                           title='Events Data - question features', save_name='questions_event_features .png',  OVERRIDE=OVERRIDE)
        elif is_plot_by_columns:
            simple_df_plot(question_features,
                           columns=['n_students', 'n_correct_answers', 'n_wrong_answers', 'percent_wrong_answers'],
                           kind = 'bar', is_legend = False, is_subplots = True, reset_index = True,
                           title='Events Data - basic question features',
                           save_name='questions_event_features- basic.png',
                           OVERRIDE=OVERRIDE)
            simple_df_plot(question_features, columns=['top0.30', 'top0.50', 'top0.80'],
                           kind='bar', is_legend = True, is_subplots = False, reset_index = True, figsize=(15,5),
                           title='Events Data - precentile question features',
                           save_name='questions_event_features-percentile.png',
                           OVERRIDE=OVERRIDE)
            simple_df_plot(question_features, columns=['mistakes_normalized_entropy'],
                           kind = 'bar', is_legend = True, is_subplots = False, reset_index = True, figsize=(15,5),
                           title='Events Data - entropy', save_name='questions_event_features-normalized_entropy.png',
                           OVERRIDE=OVERRIDE)
            simple_df_plot(question_features,
                           columns=['n_common_mistakes(>0.10)', 'n_non_common_mistakes (<0.01 or 1 student)'],
                           kind = 'bar', is_legend = True, is_subplots = False, reset_index = True, figsize=(15,5),
                           title='Events Data - advanced question features',
                           save_name='questions_event_features-common vs non common count.png', OVERRIDE=OVERRIDE)
            simple_df_plot(question_features,
                           columns=['percent_common_mistakes(>0.10)', 'percent_non_common_mistakes (<0.01 or 1 student)'],
                           kind='bar', is_legend=True, is_subplots=False, reset_index=True, figsize=(15, 5),
                           title='Events Data - advanced question features',
                           save_name='questions_event_features-common vs non common percent.png',
                           OVERRIDE=OVERRIDE)

        return question_features


class SystematicEventAnalyzer(QuestionEventAnalyzer):

    def pairwise_question_comparison(self, responses_to_question_1, responses_to_question_2):
        clean_response= lambda s: s[s.find(': ')+3:-2].replace("\\\\frac","").replace("}{","/")
        self.test_results_items = ['mi', 'chi2','mi_fixed', 'chi2_fixed', 'p', 'dof', 'n_students', 'n_joint_responses',
                                   'n_students_per_joint_responses',
                                   'max_n_students_joint_responses', 'joint_responses_dict']
        empty_response=[pd.Series([-1 for i in self.test_results_items],index=self.test_results_items),df(),df()]
        if len(responses_to_question_1)==0 or len(responses_to_question_2)==0:
            return empty_response


        responses=df(pd.concat([responses_to_question_1, responses_to_question_2],axis=1)) #remove students who didn't answer one of the questions.
        responses.dropna(how='any', inplace=True)
        n_students=len(responses)
        if len(responses)<2:
            return empty_response #no joint students
        else:
            try:
                joint_responses=pd.Series(list(zip(responses[responses_to_question_1.name], responses[responses_to_question_2.name])), index=responses.index)
            except ValueError:
                print(responses)
                return empty_response

            observed_counts=joint_responses.value_counts()
            observed_f=observed_counts/len(joint_responses)
            observed_f.index=pd.MultiIndex.from_tuples(observed_f.index)
            observed_f=observed_f.unstack().fillna(0.)
            observed_counts.applymap(int)
            chi2, p, dof, expected_f = chi2_contingency(observed_f)
            g, pg, dofg, expected = chi2_contingency(observed_f, lambda_="log-likelihood")

            mi = 0.5 * g / observed_f.unstack().sum()

            observed_counts_filtered=observed_counts[observed_counts>1]
            joint_responses_dict=str(dict(observed_counts_filtered))
            n_joint_responses=len(observed_counts_filtered)
            if n_joint_responses>0:
                mi_fixed=mi
                chi2_fixed=chi2
            else:
                chi2_fixed = 0
                mi_fixed = 0
            n_students_per_joint_responses=observed_counts_filtered.sum()
            max_n_students_joint_responses=observed_counts_filtered.max()
            expected_f=df(expected_f, index=observed_f.index, columns=observed_f.columns)
            #save results to df:
            test_results=pd.Series([mi, chi2 ,mi_fixed, chi2_fixed, p ,dof ,n_students, n_joint_responses,n_students_per_joint_responses,max_n_students_joint_responses,joint_responses_dict],
                                   index=self.test_results_items)
            #return chi2, p, dof, observed_f, expected_f, n_students, mi , responses_dict
            return test_results, observed_f, expected_f





    def running_pairwise_question_comparison(self, questions_list, how='all_vs_all', is_plot=False, is_save_to_csv=False, print_details=False):
        question_id1 = questions_list[0]
        responses1, marginal_frequency1 = self.get_response_vector(question_id1,clean_response=True)
        results_items=['all']#['chi2', 'p', 'dof', 'n_students','mi']
        results_table=df(index=questions_list, columns=results_items)
        #\todo - continue here - make running window over question
        for q_ind in range(len(questions_list)-1):
            question_id2=questions_list[q_ind+1]
            responses2, marginal_frequency = self.get_response_vector(question_id2 , clean_response=True)
            test_results, observed_f, expected_f =self.pairwise_question_comparison(responses1, responses2)


            if print_details:
                print(df(list(set(responses1.index).intersection(set(responses2.index))), columns=['---joint students:---']))
                print('----responses 1:----')
                print(responses1)
                print('----responses 2:----')
                print(responses2)
                print('----observed:----')
                print(observed_f)
                print('----expected:----')
                print(expected_f)
                #print('chi2=%s , p=%s, dof=%s, n_students=%s' %(chi2, p, dof, n_students))
            if results_items==['all']:
                results_table.columns=list(test_results.index)
                results_items=list(test_results.index)
            results_table[results_items].loc[question_id2]= test_results[results_items]
            responses1=responses2.index


                #results_table.loc[question_id2][['chi2', 'p', 'dof', 'n_students']] = df([chi2, p, dof, n_students])


        #add meta data to results table:
        print(self.meta_data.columns)
        md_index=self.meta_data[['LO_subject_index', 'LO_subsubject_index', 'LO_general_index','LO_combined_index','num_of_questions_in_lo_session','question_index']]
        md_index.index=self.meta_data.sElementID
        questions_md=md_index.loc[questions_list].drop_duplicates()
        results_table[questions_md.columns]=questions_md

        self.chi2_restuls_table=results_table



        if is_plot:
            simple_df_plot(results_table,is_subplots=True, reset_index=True, is_legend=True, save_name='chi2_restuls_all_questions.png',OVERRIDE=True)

        if is_save_to_csv:
            results_table.to_csv(os.path.join(OUTPUT_DIR,'chi2_running_pairwise_results.csv'))


        return results_table

    def all_vs_all_comparison(self, questions_list, is_plot=False, is_save_to_csv=False):

        results_items=['all']#['chi2', 'p', 'dof', 'n_students','mi']
        print('------------len questions list=%s-----------------' %len(questions_list))
        res = {}
        responses={}
        calculated_questions=[]
        for q1 in questions_list:
            if q1 in responses.keys():
                responses1=responses[q1]
            else:
                responses1, marginal_frequency1 = self.get_response_vector(q1,clean_response=True)
                responses[q1]=responses1
            calculated_questions.append(q1)
            print('q1:%s' %q1)
            print('q1:%s vs. %s' %(len(calculated_questions),len(questions_list)-len(calculated_questions)))
            for q2 in questions_list:
                if q2 not in calculated_questions:
                    if q2 in responses.keys():
                        responses2 = responses[q2]
                    else:
                        if q2=='question_e83cee06-062d-4226-b5e0-0769ea5c9832':
                            print('break')
                        responses2, marginal_frequency2 = self.get_response_vector(q2,clean_response=True)
                        responses[q2] = responses2

                    test_results, observed_f, expected_f= self.pairwise_question_comparison(responses1, responses2)
                    if results_items==['all']:
                        results_items=list(test_results.index)
                        for i in results_items:
                            res[i] = df(index=questions_list, columns=questions_list)
                    for i in results_items:
                        res[i][q1].loc[q2] = test_results[i]

        if is_save_to_csv:
            full_results=df(index=res[results_items[0]].unstack().index,columns=results_items)

            meta_data_columns=['LO_subject_index', 'LO_subsubject_index', 'LO_general_index', 'LO_combined_index',
                 'num_of_questions_in_lo_session', 'question_index']
            md = self.meta_data[meta_data_columns
                ].copy()
            md.index = self.meta_data.sElementID
            md.drop_duplicates(inplace=True)
            questions_md = md.loc[questions_list].drop_duplicates()
            for i in results_items:
                res[i].to_csv(os.path.join(OUTPUT_DIR,'all_vs_all_chi2_results_%s.csv' %i))
                full_results[i]=res[i].unstack()
            md_q1 = md.loc[full_results.index._get_level_values(0)]
            md_q1.columns=['%s_1' %c for c in meta_data_columns]
            md_q1.index=full_results.index
            md_q2= md.loc[full_results.index._get_level_values(1)]
            md_q2.columns = ['%s_2' % c for c in meta_data_columns]
            md_q2.index = full_results.index

            full_results=pd.concat([full_results,md_q1,md_q2],axis=1)
            full_results['same_LO']=full_results.LO_general_index_1 == full_results.LO_general_index_2
            full_results['same_subject'] = full_results.LO_subject_index_1 == full_results.LO_subject_index_2

            full_results.to_csv(os.path.join(OUTPUT_DIR,'all_vs_all_chi2_results_full.csv'))
        return res, full_results

    def select_questions(self):
        responses2, marginal_frequency2 = self.get_response_vector(question_id2)
        self.preprocess_event_data(filter_only_first_attempt=True)
        question_indeces=self.processed_event_data
        return


"""@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""


