% query_outcome(1). query_condition(1, 1).query_condition(2, 0). query_exception(1, 0).query_exception(2, 1).

yes :- query_outcome(1), not query_exception(_, 1), not query_condition(_, 0).

yes :- query_outcome(0), not query_exception(_, 0), not query_condition(_, 1).

no :- not yes.

#show yes/0.
#show no/0.