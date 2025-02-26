



% query entail bfactor(X), node bfactor(X) is activated
bfactor(X) :- entail(query, bfactor(X)).

% sup/atk relationship from X to T is activated if there are link between them and X is activated
support(X,Y) :- link(support(X,Y)), X.
attack(X,Y) :- link(attack(X,Y)), X.

% node afactor(Y) is activated if there is an actived support link to afactor(Y) and there is no actived attack link.
afactor(Y) :- support(_,afactor(Y)), not attack(_,afactor(Y)).

% root(X) is activated if there is an actived support link to root(X) and there is no actived attack link.
root(X) :- support(_,root(X)), not attack(_, root(X)).


yes :- root(X).

no :- not yes.

#show yes/0.
#show no/0.