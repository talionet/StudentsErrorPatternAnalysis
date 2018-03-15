from analysis import event_analysis_by_student
from settings import *


cluster_by= 'num_joint_errors'
global output_folder
output_folder=os.path.join(OUTPUT_DIR,'Students_cluster_analysis')

def main():
    SEA=event_analysis_by_student.StudentEventAnalyzer()
    SEA.cluster_students_by_responses(cluster_by=cluster_by, min_joint_questions=10,
                                              one_section_questions_only=True, load_raw_data=False,
                                              load_processed_data=True,
                                              processed_data_file_name='events_table_math_processed_first_attempt.csv',
                                              load_similarity_matrix=False, is_plot=True)
    SEA.describe_students_clusters(is_plot=True)