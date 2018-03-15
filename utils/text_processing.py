def raw_event_response_cleaner(raw_response):
    """
    processing of ‘Response’ field by  raw_event_response_cleaner.py to obtain a str which contains only students response
    (clean response field), when there is more than one section this returns a dict.
    """
    clean_response = {}

    if type(raw_response)==float:
        return {}

    elif raw_response[0]=='t':
        for r in raw_response.split(';'):
            q = r[:r.find(':')].replace(' ', '') #question key
            responses=r[r.find(':')+2:]
            if len(q)>0:
                clean_response[q] = responses

    elif raw_response[0]=='e':
        for r in raw_response.split(';'):
            q=r[:r.find(':')]
            q=q.replace(' ','')
            responses=r[r.find('"editable":')+11:r.find(']')+1]
            responses=responses.replace('"latex":','').replace('"','').replace('[','').replace(']','')
            clean_response[q]=responses

    return clean_response



