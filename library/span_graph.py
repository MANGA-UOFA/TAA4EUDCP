import numpy as np


class SpanGraph:
        
    def continuous_spans_are_compatible(span1, span2):
        span1_begin, span1_end = span1
        span2_begin, span2_end = span2
        return (
            ((span1_begin <= span2_begin) and (span1_end >= span2_end)) or \
            ((span2_begin <= span1_begin) and (span2_end >= span1_end)) or \
            (span1_end <= span2_begin) or (span2_end <= span1_begin)
        )
    
    def continuous_discontinuous_spans_are_compatible(continuous, discontinuous):
        continuous_begin, continuous_end = continuous
        discontinuous_left_begin, discontinuous_left_end, discontinuous_right_begin, discontinuous_right_end = discontinuous
        return (
            ((continuous_begin <= discontinuous_left_begin) and (continuous_end >= discontinuous_right_end)) or \
            ((discontinuous_left_begin <= continuous_begin) and (discontinuous_left_end >= continuous_end)) or \
            ((discontinuous_right_begin <= continuous_begin) and (discontinuous_right_end >= continuous_end)) or \
            (continuous_end <= discontinuous_left_begin) or (continuous_begin >= discontinuous_right_end) or \
            ((continuous_end <= discontinuous_right_begin) and (continuous_begin >= discontinuous_left_end))
        )
    
    def discontinuous_spans_are_compatible(spanA, spanB):
        def ordered_discontinuous_spans_are_compatible(span1, span2):
            span1_left_begin, span1_left_end, span1_right_begin, span1_right_end = span1
            span2_left_begin, span2_left_end, span2_right_begin, span2_right_end = span2
            return (
                ((span1_left_begin <= span2_left_begin) and (span1_left_end >= span2_left_end) and (span1_right_begin <= span2_right_begin) and (span1_right_end >= span2_right_end)) or \
                ((span1_left_begin <= span2_left_begin) and (span1_left_end >= span2_right_end)) or \
                ((span1_right_begin <= span2_left_begin) and (span1_right_end >= span2_right_end)) or \
                ((span1_left_end <= span2_left_begin) and (span1_right_begin >= span2_left_end) and (span1_right_end <= span2_right_begin)) or \
                ((span1_right_end <= span2_left_begin)) or \
                ((span1_left_end <= span2_left_begin) and (span1_right_begin >= span2_right_end))
            )
        return ordered_discontinuous_spans_are_compatible(spanA, spanB) or ordered_discontinuous_spans_are_compatible(spanB, spanA)

        
    def compatible_spans(span1, span2):
        if span1==span2:
            return True
        if len(span1)==len(span2)==2:
            return SpanGraph.continuous_spans_are_compatible(span1, span2)
        if len(span1)==len(span2)==4:
            return SpanGraph.discontinuous_spans_are_compatible(span1, span2)
        if len(span1)==2 and len(span2)==4:
            return SpanGraph.continuous_discontinuous_spans_are_compatible(span1, span2)
        if len(span1)==4 and len(span2)==2:
            return SpanGraph.continuous_discontinuous_spans_are_compatible(span2, span1)
        raise Exception("Wrong spans!")
                
        
    def __init__(self, hit_counts):
        self.size = len(hit_counts)
        self.spans = list(hit_counts.keys())
        self.span2idx = {span: i for i, span in enumerate(self.spans)}
        self.matrix = np.zeros((self.size, self.size))
        self.scores = np.array([hit_counts[span] for span in self.spans])
        for this_span in self.spans:
            for other_span in self.spans:
                if SpanGraph.compatible_spans(this_span, other_span):
                    self.matrix[self.span2idx[this_span], self.span2idx[other_span]] = 1
                    self.matrix[self.span2idx[other_span], self.span2idx[this_span]] = 1

    def is_complete(self, selection):
        return np.all(self.matrix[selection][:, selection])

    def total_hitcounts(self, selection):
        return self.scores[selection].sum()