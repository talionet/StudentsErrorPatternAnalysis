from analysis.event_analysis_by_question import *

def main():
    EA=QuestionEventAnalyzer()
    min_num_students=150
    min_answers = 5
    is_save_csv=True
    is_plot=True
    EA.get_all_questions_features(questions=None, min_num_students=min_num_students, min_answers=min_answers, max_answers=200,is_save_csv=is_save_csv)
    print(EA.question_features.head())

    if is_plot:
        EA.plot_question_features(is_plot_by_columns=True, plot_columns='all', OVERRIDE=True)
