from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet,inconsistent
from scipy.spatial import distance

from agg_question_data_processing import *
from analysis.event_analysis import EventsAnalyzer
from utils import data_handle
from visualization import *

#quick reminder about clustering analysis:
#https://www.mathworks.com/help/stats/hierarchical-clustering.html
EA=EventsAnalyzer()

class StudentEventAnalyzer(EventsAnalyzer):
    """performs students clustering analysis by their response to CET questions. """

    def cluster_students_by_responses(self, cluster_by='correct_incorrect', min_joint_questions=10, fcluster_criteria=0.8,
                                          one_section_questions_only=True, load_raw_data=False,
                                          load_processed_data=True,
                                          processed_data_file_name='events_table_math_processed_first_attempt.csv',
                                          load_similarity_matrix=False, is_plot=True):

        """
        :param cluster_by:
            'sum_joint_questions' - cluster students by the questions they answered regardless of response
            'correct_incorrect' - cluster by the number of errors divided by the number of joint responses (questions answered by both students)
            'percent_joint_errors' - cluster by the percent of joint errors (questions both students answered incorrectly) which are joint for both students

        :param min_joint_questions: minimal number of questions both students answered in order to calculate distance
        :param load_raw_data:
        :param load_processed_data:
        :param processed_data_file_name:
        :param load_similarity_matrix:
        :param is_plot:
        :return:
        """
        # ---- load events data and meta data
        EA.load_data(load_raw_data=load_raw_data, load_processed_data=load_processed_data,
                     processed_data_file_name=processed_data_file_name)

        data = EA.processed_event_data
        md = EA.meta_data.set_index('sElementID')

        # ----- define similarity matrix properties:
        if cluster_by == 'sum_joint_questions':
            data_column = 'is_correct_response' # which columns from raw data to use
            mask_type = 'bool' # apply mask over data columns
            metric_type = 'hamming'  # type of distance for pairwise dist for linkage calculation
            linkage_type = 'complete' #linkage for pairwise clustering
            fillna = None

        elif cluster_by == 'correct_incorrect':
            data_column = 'is_correct_response'
            mask_type = 'nan_to_zero'
            metric_type = 'jaccard_intersection'  # todo consider normalized measures\ give higher score to mistakes.
            linkage_type = 'average'
            distance_normalization='min_max'
            fillna = 'max_and_std'

        elif cluster_by == 'percent_joint_errors': # todo - define this and check.
            data_column = 'is_correct_response'
            mask_type = 'nan_to_zero' #replace 0 to -1 and nan to zero
            metric_type = 'joint_minus_percent_drop0'
            linkage_type = 'average'
            fillna = 'max'
            distance_normalization = None

        elif cluster_by == 'num_joint_errors': # todo - define this and check.
            data_column = 'is_correct_response'
            mask_type = 'nan_to_zero'
            # calcs the mean correct\incorrect response over joiint questions
            metric_type = 'joint_minus_count_drop0'
            linkage_type = 'average'

            distance_normalization='min_max'
            fillna = 'max'

        metric_name = metric_type
        if metric_type not in dir(distance):
            metric_type = data_handle.define_distance_metric(distance_name=metric_type,
                                                             min_intersection=min_joint_questions)

        #self.clustering_details = utils.make_dict_from_locals(locals(), keys=
            #['data_column', 'mask_type', 'metric_type', 'fillna', 'linkage_type'])  # the column used for clustering


        if one_section_questions_only:
            data = data.loc[data.n_sections == 1]
        # ---- load similarity matrix according to metric name
        if load_similarity_matrix:
            students_similarity_matrix = df.from_csv('temp_similarity_matrix_%s_%s_metric_normalized_%s.csv' % (mask_type, metric_name,str(distance_normalization)))
            if students_similarity_matrix.shape[1]==1:
                students_similarity_matrix=students_similarity_matrix.T.values[0]
            elif students_similarity_matrix.shape[1]>1:
                students_similarity_matrix=distance.squareform(students_similarity_matrix.values)
            processed_students_responses=df.from_csv('%s_students_responses.csv' % mask_type)


        else:

            process_for_figure = True

            students_responses=data_handle.pivot(data, index_col='student_id',columns_col='question_id', values_col=data_column,
                                   agg_function='first',convert_to_numeric=True)


            #-------------------------------
            processed_students_responses = data_handle.mask_data(students_responses, type=mask_type)
            print(processed_students_responses)
            #students_similarity_matrix10 = distance.pdist(processed_students_responses.head(10), metric=metric_type)
            students_similarity_matrix=distance.pdist(processed_students_responses, metric=metric_type)
            df(distance.squareform(students_similarity_matrix), index=processed_students_responses.index,
               columns=processed_students_responses.index).to_csv(
                'temp_similarity_matrix_%s_%s_metric_NOT_Normalized.csv' % (mask_type, metric_name))
            students_similarity_matrix=data_handle.normalize_data(students_similarity_matrix,min_value=0, by=distance_normalization, fillna=fillna)
            studetns_similarity_matrix=students_similarity_matrix[0]
            df(distance.squareform(students_similarity_matrix), index=processed_students_responses.index,
               columns=processed_students_responses.index).to_csv(
                'temp_similarity_matrix_%s_%s_metric_normalized_min_max.csv' % (mask_type, metric_name))
            df(processed_students_responses).to_csv('%s_students_responses.csv' % mask_type)



        #def hierarhical_clustering(smilarity_matrix, linkage_method):
        #students_similarity_matrix=df.from_csv('temp_similarity_matrix_100.csv')
        Z=linkage(students_similarity_matrix, method=linkage_type)
        Z_df=df(Z,columns=['obs1','obs2','distance','n_in_cluster'])
        c=cophenet(Z,students_similarity_matrix)
        print('cophenet=%f for linkage==%s  and distance==%s' %(c[0],linkage_type, metric_name))
        inconsistency= df(inconsistent(Z,), columns=['mean_link','std_link','n_links','inconsistence_coeff'])
        students_clusters=pd.Series(fcluster(Z, fcluster_criteria))#5, criterion='maxclust')
        print(students_clusters.value_counts())
        cluster_responses=processed_students_responses.copy()


        cluster_responses['cluster']=students_clusters
        cluster_responses.sort_values('cluster', inplace=True)
        questions_count_by_cluster=df(columns=set(students_clusters))

        #show the dendrogram next to the feature matrix to check if it makes sense
        if is_plot:
            cr=cluster_responses.T.apply(pd.to_numeric).drop('cluster')
            f,axes=plt.subplots(2,1,sharex=True)
            plt.subplot(211)
            plt.title('clustering by students response %s' %cluster_by)
            plt.ylabel('distance')

            dendrogram(Z,no_labels=True, color_threshold=0.7*max(Z[:,2]))
            plt.subplot(212)
            cr.index=md.loc[cr.index]['question_index'].drop_duplicates()
            cr.sort_index(inplace=True)
            #cr=cr.applymap(lambda x : np.nan if x==0 else x).dropna(how='all',axis=0)

            plt.pcolor(cr)

            plt.ylabel('question')
            plt.xlabel('student')
            #plt.plot(kind='bar')
            plt.savefig('heatmap_and_dendrogram_%s.png' % cluster_by)
            plt.close()

        self.students_clusters=students_clusters
        self.clustered_students_responses=cluster_responses
        self.students_similarity_matrix=students_similarity_matrix
        return students_clusters, cluster_responses, students_similarity_matrix

    def describe_students_clusters(self, min_in_cluster=10, is_plot=False,columns=['n_students', 'n_questions (>1)', 'n_questions (>20)','n_students in questions (>20)']):
        #analyze the clusters recieved:

        #remove clusters with less than minimum number of members
        common_clusters = self.students_clusters.value_counts()[self.students_clusters.value_counts() > min_in_cluster].index
        students_clusters = self.students_clusters.apply(lambda x: x if x in common_clusters else 0)

        clusters=set(students_clusters)
        questions_count_by_cluster=df(columns=clusters)
        students_description=df(index=clusters, columns=columns)
        students_description['n_students']=students_clusters.value_counts()
        for c in clusters:
            cluster_index=students_clusters[students_clusters == c].index
            cluster_students = self.clustered_students_responses.iloc[cluster_index].drop('cluster',axis=1)
            n_students_per_question=cluster_students.sum()
            questions_count_by_cluster[c]=((cluster_students==-1)).sum()

            if 'n_questions (>1)' in columns:
                students_description['n_questions (>1)'].loc[c]=len(n_students_per_question[n_students_per_question>1])
            if 'n_questions (>20)' in columns:
                students_description['n_questions (>20)'].loc[c]=len(n_students_per_question[n_students_per_question > 20])
            if 'n_students in questions (>20)' in columns:
                students_description['n_students in questions (>20)'].loc[c] = sum(n_students_per_question[n_students_per_question > 20])
            if 'n_errors' in columns:
                cluster_errors = (cluster_students<0).sum()
                relevant_questions=cluster_errors[cluster_errors>1].index
                cr=cluster_students[relevant_questions]
                plt.pcolor(cr)
                plt.colorbar()
                n_questions_error = (cluster_errors >1).sum()
                errors= (cr == -1) * 1.
                total_answers= (cr !=0) *1.
                #how many students had mistaken in each question
                question_error_percent =errors.sum()/total_answers.sum()
                #how many errors did each student have
                students_error_percent= errors.T.sum()/total_answers.T.sum()


        if is_plot:
            cluster_errors = (cluster_students < 0).sum()
            relevant_questions = cluster_errors[cluster_errors > 1].index
            cr = cluster_students[relevant_questions]
            plt.pcolor(cr)
            plt.title('students response to questions in cluster')
            plt.xlabel('question')
            plt.ylabel('student')
            plt.savefig('clustered_students_responses.png')
            plt.close()

        #arrange data by LOs:

        md=EA.meta_data.set_index('sElementID')
        questions_count_by_cluster.index=md.loc[questions_count_by_cluster.index]['question_index'].drop_duplicates()
        questions_count_by_cluster.sort_index(inplace=True)
        if is_plot:
            questions_count_by_cluster.plot(kind='bar',subplots=True, sharex=True, title=['']*len(clusters),use_index=False)
            plt.savefig('questions_by_clusters.png')

        #md=EA.meta_data[['LO_subject_index', 'LO_subsubject_index','LO_general_index','question_index']]

        #print(students_similarity_matrix)





