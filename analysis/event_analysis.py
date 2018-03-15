from agg_question_data_processing import *
from utils.text_processing import raw_event_response_cleaner
from visualization import *


class EventsAnalyzer():
    """
    analysis of CET raw events data (returned from .json by CetEventProcessing --> Event_Processing.CetEventProcessing.load())

    """
    def __init__(self):
        self.event_data = None
        self.meta_data = None
        self.processed_event_data = None
        self.top_questions = None
        return

    def load_data(self, load_raw_data=True,
                  load_processed_data=False,
                  load_meta_data=True,
                  agg_data_file_name='question_session_log_13.csv',
                  processed_data_file_name='ENTER_IN_MAIN_processed_data_file_name',
                  event_data_file_name='events_table_math.csv',
                  meta_data_file_name='MD_math_processed.csv',
                  ):
        #event_data_file_name = 'events_table_full.csv'

        if load_raw_data:
            print('loading event data from %s...' %os.path.join(DATA_ROOT,event_data_file_name))
            self.event_data=df.from_csv(os.path.join(DATA_ROOT,event_data_file_name))
            print(self.event_data.columns)
            '''Index(['event_id', 'question_id', 'time', 'student_id', 'object_url', 'lo_id',
                   'session_id', 'page_url', 'action', 'completion', 'score', 'response',
                   'clean_response', 'full_or_partial_success', 'full_attempts',
                   'full_attempts_desc', 'full_answer', 'n_sections'],
                  dtype='object')'''

        if load_processed_data:
            print('loading event data from %s...' %os.path.join(DATA_ROOT,event_data_file_name))
            self.processed_event_data=df.from_csv(os.path.join(DATA_ROOT,processed_data_file_name))
            print(self.processed_event_data.columns)
            '''Index(['event_id', 'time', 'student_id', 'object_url', 'lo_id', 'session_id',
               'page_url', 'action', 'question_id', 'completion', 'score', 'response',
               'full_or_partial_success', 'full_attempts', 'full_attempts_desc',
               'n_attempt', 'clean_response', 'n_sections'],
                  dtype='object')'''

        if load_meta_data:
            print('loading event data from %s...' % os.path.join(DATA_ROOT, meta_data_file_name))
            self.meta_data=df.from_csv(os.path.join(DATA_ROOT,meta_data_file_name))
            '''Index(['sElementID', 'gLO', 'nVersion', 'sName', 'nQuestionIndex',
                   'sQuestionPageID', 'nLanguage', 'dtCreatedDate', 'sSyllabus', 'sLOurl',
                   'nPages', 'sQtype', 'sSubjects', 'sSubSubjects', 'sDetail', 'sGoals',
                   'sRepresentation', 'sActivityType'],'''



    def preprocess_event_data(self, load_processed_data=False, processed_data_file_name=os.path.join(DATA_ROOT,'processed_data','events_table_math_processed_first_attempt.csv'),
           filter_by_meta_data=True, filter_only_single_section_data=True, add_response_descriptors=True,
                              filter_only_first_attempt=True, filter_by_min_n_students=3,
                              filter_by_min_n_answers=False, filter_by_language=True, arrange_by_index=True,
                              add_error_type_to_data=False,clean_raw_response=True, drop_non_informative_responses=False):
        """
        - Adds 'n_attempt' field to data (number of attempt at a specific question for each student)
        - filter data by 'asked_check' and by params.
        :param
        - filter_only_single_section_data: if True -- use_only questions with single section
        - filter_only_first_attempt: if True -- use only the first attempt of the student in each question
        - filter_by_min_n_students: use only questions answered by min num of students - optional
        - filter_by_min_n_answers: se only questions which had min number of unique responses - optional

        :return:
        processed and filtered data
        """
        if load_processed_data:
            self.is_load_raw_data=False
            self.load_data(load_raw_data=False, load_processed_data=True,
                           processed_data_file_name=processed_data_file_name)
            data=self.processed_event_data

        else:
            #preprocess the raw data_using deafult filters:
            if type(self.event_data)!= pd.core.frame.DataFrame:
                self.load_data()

            data=self.event_data.loc[self.event_data.action=='asked_check']
            print('filtering and processing data...(N=%i)' % len(data))

            if filter_by_meta_data:
                questions_list_md=set(data['question_id']).intersection(set(self.meta_data.sElementID))
                data_in_md_index=[i for i in data.index if data.question_id.loc[i] in questions_list_md]
                data=data.loc[data_in_md_index]
                print('filtering only questions in meta data...(N=%i)' % len(data))

            if filter_by_language:
                filtered_questions=self.meta_data.loc[self.meta_data.nLanguage==1].sElementID.values
                filtered_index = [i for i in data.index if data.question_id[i] in filtered_questions]
                data = data.loc[filtered_index]
                print('filtered nLanguage=1 only... (N=%i)' %len(data))

            data=data.loc[data.full_attempts == 1.]
            data['n_attempt'] = data.groupby(['question_id','student_id']).full_attempts.cumsum()

            if filter_only_first_attempt:
                print('filter first attempt only...(N=%i)' %len(data))
                data=data.loc[data.n_attempt==1]
                data.drop_duplicates(['question_id', 'student_id'], keep='first', inplace=True) #drops questions answered by the same students more than once

            if clean_raw_response:
                # raw_cleaner = lambda s: s[s.find(': ') + 3:-2].replace("\\\\", "").replace("frac", "").replace("}{", "/")
                clean_responses = data['response'].apply(raw_event_response_cleaner)
                data['clean_response'] = clean_responses
                # data['resoponse_type']=clean_responses.apply(lambda a:a[0] if len(a)>0 else np.nan)
                data['n_sections'] = data['clean_response'].apply(len)


            if add_response_descriptors:
                data['is_correct_response'] = self.calc_response_correctness(data)
                response_student_percent,response_student_count=self.calc_response_commonality(data)
                data['response_percent']=response_student_percent
                data['response_count']=response_student_count
                data.to_csv('processed_data_temp.csv')


        #apply data filters on processed data:
        if drop_non_informative_responses:
            excluded_responses = ['{}']
            excluded_charts = ['&', '=', '?']

            data = data[~data['clean_response'].apply(lambda i: i in excluded_responses)]
            data = data[~data['clean_response'].apply(lambda i: any([c in i for c in excluded_charts]))]
            print('filtered data by excluded charts %s and responses %s... (N=%i)' % (
            excluded_responses, excluded_charts, len(data)))

        if type(filter_by_min_n_students)==int:
            n_students_per_question=data.groupby('question_id')['student_id'].value_counts().groupby('question_id').count()
            filtered_questions=n_students_per_question.loc[n_students_per_question >= filter_by_min_n_students].index
            filtered_index=[i for i in data.index if data.question_id[i] in filtered_questions]
            data=data.loc[filtered_index]
            print('filter n_students>%i only... (N=%i)' % (filter_by_min_n_students,len(data)) )


        if type(filter_by_min_n_answers)==int:

            n_students_per_question = data.groupby('question_id')['response'].value_counts().groupby('question_id').count()
            filtered_questions = n_students_per_question.loc[n_students_per_question >= filter_by_min_n_answers].index
            filtered_index = [i for i in data.index if data.question_id[i] in filtered_questions]
            data = data.loc[filtered_index]
            print('filter n_answers>%i only... (N=%i)' %(filter_by_min_n_answers,len(data)))


        if filter_only_single_section_data:
            if 'n_sections' not in data.columns:
                data_with_response=data['full_attempts_desc'].dropna()
                n_sections=data_with_response.apply(lambda s: len(eval(s)))
                data=pd.concat([data, n_sections],axis=0)
            data = data.loc[data.n_sections == 1]
            #fix questions with n_section ==1 and full_attempts==0 (happens when there is a 'explain' window
            print('filter n_sections==1 only... (N=%i)' %len(data))



        if add_error_type_to_data: #use in case manual labels of responses errors were obtained
            data=self.add_error_type_to_event_table(data)

        if arrange_by_index:
            data.reset_index(inplace=True,drop=True)
            question_indexes = self.meta_data[['LO_subject_index', 'LO_subsubject_index',
                                              'LO_general_index', 'LO_combined_index', 'question_index']].copy()
            question_indexes.index=self.meta_data.sElementID
            question_indexes.drop_duplicates(inplace=True)

            data[question_indexes.columns]=question_indexes.loc[data.question_id].reset_index(drop=True)

            data.sort_values(by='question_index',inplace=True)

        self.processed_event_data = data

        #arrange questions list by LO and questions indexes
        questions_list=list(set(self.processed_event_data.question_id).intersection(set(self.meta_data.sElementID )))
        questions_list_ordered = question_indexes.loc[questions_list]
        questions_list_ordered=questions_list_ordered['question_index'].sort_values().dropna().index
        self.processed_questions_list=list(questions_list_ordered)
        self.question_indexes=question_indexes
        print('number of questions = %i' % len(self.processed_questions_list))


    #describe events per question to find interesting questions:
    def agg_by_question(self, is_plot=False, min_num_students=150, min_answers=3, max_answers=200, n_questions='all'):
        """ For each question, calculates the number of students who answered it, the number of events (studentsXresponses) and the number of unique answers """
        question_description = df()
        question_description['students_per_question'] = self.processed_event_data.groupby('question_id')['student_id'].value_counts().groupby('question_id').count()
        question_description['events_per_question'] = self.processed_event_data.groupby('question_id')['event_id'].value_counts().groupby('question_id').count()
        question_description['answers_per_question'] = self.processed_event_data.groupby('question_id')['clean_response'].value_counts().groupby('question_id').count()
        #number_of_responses_per_question=single_section_data.groupby('question_id')['clean_response'].value_counts()

        question_description.sort_values('students_per_question',ascending=False,inplace=True)

        #filter ineresting questions:
        interesting_questions= question_description.loc[question_description.students_per_question>min_num_students].loc[question_description.answers_per_question>min_answers].loc[question_description.answers_per_question<max_answers]
        interesting_questions.sort_values('answers_per_question',ascending=False,inplace=True)
        """Index(['question_25504e71-8563-4600-aa5e-be589b5e44c5',
       'question_b39b61da-cd9c-4805-81fd-df32c4affb97',
       'question_d20b14ef-b0ae-4d10-8a33-83246de7af84',
       'question_d9bc9247-5649-42d5-bc73-c6794c3cb640',
       'question_8af33bc4-0084-466e-8ab1-08448f4f277e'])"""
        if is_plot:
            simple_df_plot(question_description, title='events - question description', save_name='questions_event_description.png', reset_index=True, OVERRIDE=True)

        self.question_description = question_description
        if n_questions == 'all':
            self.top_questions = interesting_questions
        else:
            self.top_questions=interesting_questions.iloc[:min(n_questions,len(interesting_questions))]

        return self.top_questions.index

    def agg_by_question_attempts(self,only_top_questions=True,is_plot=False):
        if type(self.event_data)!= pd.core.frame.DataFrame:
            self.load_data()
        if only_top_questions:
            if not self.top_questions:
                self.agg_by_question()

                n_attempts_per_question = \
                self.processed_event_data.groupby('question_id')[
                    'n_attempt'].value_counts().unstack().loc[self.top_questions.index]
                n_attempts_per_question.index=range(len(n_attempts_per_question))
                n_attempts_per_question.index+=1

        if is_plot:
            simple_df_plot(n_attempts_per_question.T, kind='bar', is_subplots=False, is_legend=False, figsize=(15, 5),
                           xlabel='attempt', ylabel='count',
                           title='events - question attempts description',
                           save_name='questions_event_attempts_description.png', OVERRIDE=True)
            simple_df_plot(n_attempts_per_question.T.iloc[1:].T, kind='bar', is_subplots=False, is_legend=False, figsize=(15, 5),
                           xlabel='attempt (>1)', ylabel='count',
                           title='events - question attempts description',
                           save_name='questions_event_attempts_description.T.png',  OVERRIDE=True)

        return n_attempts_per_question

    def calc_response_commonality(self, processed_event_data):
        response_percent = pd.Series(index=processed_event_data.index)
        response_count = pd.Series(index=processed_event_data.index)

        response_counts_df=processed_event_data.groupby(['question_id','clean_response']).student_id.count().unstack()

        for question in response_counts_df.index:
            question_responses=response_counts_df.loc[question].T.dropna()
            if len(question_responses)>0:
                question_events= processed_event_data.loc[processed_event_data.question_id == question]['clean_response']
                n_responses=len(question_events)
                for response in question_responses.index:
                    question_response_count=question_responses[response]
                    question_response_index=question_events.loc[question_events==response].index
                    response_percent[question_response_index]=question_response_count/n_responses
                    response_count[question_response_index]=question_response_count


        return response_percent,response_count



    def calc_response_correctness(self, processed_event_data):
        """ This function checks for a given response if it's correct using the 'full_attempts description' field in raw_data and adds it to data"""
        def is_correct(attempt_desc):
            attempts=attempt_desc.split('(')[1:]
            if len(attempts)==0:
                return np.nan
            elif len(attempts)>1:
                return [int(i[0]) for i in attempts]
            else:
                return int(attempts[0][0])


        is_correct_response= processed_event_data.full_attempts_desc.apply(is_correct)
        return is_correct_response

    def add_error_type_to_event_table(self, event_data,
                                      mistakes_map_file_name='common_errors_in_LO1.csv',
                                      error_type_columns=['mistake','p','n','sQuestionPageID','sLOurl','error_type1','error_type_hebrew','error_type2','error_type2_hebrew']):
        """ in case there is manual labeling of error types, add this labeling to data"""
        #\todo - consider making this code more elegant (:
        #load file which maps response to error type and use only relevant questions:
        errors_map=df.from_csv(os.path.join(OUTPUT_DIR,mistakes_map_file_name))
        mapped_questions=set(errors_map['error_type1'].dropna().index)
        errors_map=errors_map.loc[mapped_questions].drop_duplicates()
        errors_map['mistake']=errors_map['mistake'].apply(str)
        errors_question_response_index = list(zip(errors_map.index, errors_map.mistake))
        errors_map.index=errors_question_response_index

        mapping_index=[i for i in event_data.index if event_data['question_id'].loc[i] in mapped_questions]
        mapped_events=event_data.loc[mapping_index].copy()

        #clean response data to match the responses on mapping file:
        cleaner = lambda s: s[s.find(': ') + 3:-2].replace("\\\\", "").replace("frac", "").replace("}{", "/").replace('{','').replace('}','')
        mapped_events['response_value']=mapped_events['clean_response'].apply(cleaner)
        mapped_events.index.name='Index'
        mapped_events.reset_index(inplace=True)


        event_question_response_index=list(zip(mapped_events.question_id,mapped_events.response_value))
        question_response_set=set(event_question_response_index).intersection(set(errors_question_response_index))
        mapped_events.index=event_question_response_index

        #concat both matrices and return to normal index
        mapped_events = pd.concat([mapped_events, errors_map.loc[mapped_events.index]], axis=1)
        mapped_events['question_response']=event_question_response_index
        mapped_events.index=mapped_events.Index

        return mapped_events
