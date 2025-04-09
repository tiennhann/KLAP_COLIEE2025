% inference link given examples
link(support(node(bfactor(X)), node(afactor(X)))) :- node(bfactor(X)).

node(afactor(X)) :- node(bfactor(X)).

node(root(1)).

% query entail bfactor(X), node bfactor(X) is activated
node(X, ex(EX)) :- query_entail(node(X), ex(EX)).


% there is an active support link if source node is activated and no inbound attack to that node.
support(node(X), node(Y), ex(EX)) :- link(support(node(X), node(Y))), node(X, ex(EX)), not attack(node(_), node(X), ex(EX)).


% there is an active attack link if source node is activated and no inbound attack to that node.
attack(node(X), node(Y), ex(EX)) :- link(attack(node(X), node(Y))), node(X, ex(EX)), not attack(node(_), node(X), ex(EX)).

% node is activated
node(Y, ex(EX)) :- support(node(X), node(Y), ex(EX)), node(X, ex(EX)).


% there must be an active support link to root node if root node is activated
1 <= {support(node(afactor(Y)), node(root(X)), ex(EX)): node(afactor(Y), ex(EX)) }:- root(X, true, ex(EX)).



:- root(X, false, ex(EX)), node(root(X), ex(EX)).

% if there are active support link, there must be an active source node.
node(afactor(Y), ex(EX)) :- support(node(afactor(Y)), node(root(X)), ex(EX)).

link(support(node(afactor(Y)), node(root(X)))):- support(node(afactor(Y)), node(root(X)), ex(EX)).

link(attack(node(afactor(Y)), node(afactor(X)))):- attack(node(afactor(Y)), node(afactor(X)), ex(EX)).

% % generate link type
% 1 {link(attack(node(afactor(X)), node(afactor(Y)))): node(afactor(Y)); link(support(node(afactor(X)), node(root(Z)))): node(root(Z))} :- node(afactor(X)).


% 1 {link(support(node(afactor(X)), node(root(Y)))): node(afactor(X))} :- node(root(Y)).  

0 {link(attack(node(afactor(X)), node(afactor(Y)))); link(support(node(afactor(X)), node(afactor(Y))))} 1 :- node(afactor(X)), node(afactor(Y)), X!=Y.

0 {link(support(node(afactor(X)), node(root(Y))))} 1 :- node(root(Y)), node(afactor(X)).

:- C = #count{1: link(support(node(afactor(_)), node(root(_))))}, C < 1.
% constraint on link type
:- link(attack(node(Y), node(root(X)))).
:- link(support(node(afactor(Y)), node(afactor(X)))).
:- link(attack(X,X)).
:- link(support(Y,Y)).
:- link(attack(node(afactor(Y)), node(bfactor(Z)))).
:- link(support(node(afactor(Y)), node(bfactor(Z)))).


#minimize { 1@1, L: link(L) }.

% #show node/1.
#show.
#show link(attack(node(afactor(X)), node(afactor(Y)))): link(attack(node(afactor(X)), node(afactor(Y)))).
#show link(support(node(afactor(X)), node(root(Y)))): link(support(node(afactor(X)), node(root(Y)))).
#show link(support(node(bfactor(X)), node(afactor(Y)))): link(support(node(bfactor(X)), node(afactor(Y)))).