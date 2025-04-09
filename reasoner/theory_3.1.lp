%%% -- inference label of query

% query entail bfactor(X), node bfactor(X) is activated
node(X) :- query_entail(node(X)).


% there is an active support link if source node is activated and no inbound attack to that node.
support(node(X), node(Y)) :- link(support(node(X), node(Y))), node(X), not attack(node(_), node(X)).


% there is an active attack link if source node is activated and no inbound attack to that node.
attack(node(X), node(Y)) :- link(attack(node(X), node(Y))), node(X), not attack(node(_), node(X)).

% node is activated
node(Y) :- support(node(X), node(Y)), node(X).

yes :- node(pledged), valid(1).

yes :- not node(pledged), valid(0).

no :- not yes.

#show yes/0.
#show no/0.

#show link/1.







#show node/1.
#show.
