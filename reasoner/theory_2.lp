
% query entail bfactor(X), node bfactor(X) is activated
node(X) :- entail(query, node(X)).

% sup/atk relationship from X to T is activated if there are link between them and X is activated
support(node(X), node(Y)) :- link(support(node(X), node(Y))), node(X).
attack(node(X), node(Y)) :- link(attack(node(X), node(Y))), node(X).
node(Y) :- support(_, node(Y)), not attack(_,node(Y)).

% predict label
yes :- node(root(X)), agree(query, node(root(X))).
yes :- - node(root(X)), - agree(query, node(root(X))).
no :- not yes.

#show yes/0.
#show no/0.