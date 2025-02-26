



% query entail bfactor(X), node bfactor(X) is activated
node(X) :- entail(query, node(X)).

% link(support(node(X), node(Y))) :- link(support(X,Y)).
% link(attack(node(X), node(Y))) :- link(attack(X,Y)).
% sup/atk relationship from X to T is activated if there are link between them and X is activated
support(node(X), node(Y)) :- link(support(node(X), node(Y))), node(X).
attack(node(X), node(Y)) :- link(attack(node(X), node(Y))), node(X).
node(Y) :- support(_, node(Y)), not attack(_,node(Y)).
% node afactor(Y) is activated if there is an actived support link to afactor(Y) and there is no actived attack link.


% root(X) is activated if there is an actived support link to root(X) and there is no actived attack link.


yes :- node(root(X)).

no :- not yes.

#show yes/0.
#show no/0.