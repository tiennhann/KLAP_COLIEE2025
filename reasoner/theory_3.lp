% inference link given examples
link(support(node(bfactor(X)), node(afactor(X)))) :- node(bfactor(X)).

node(afactor(X)) :- node(bfactor(X)).

node(pledged).

% query entail bfactor(X), node bfactor(X) is activated
node(X, ex(EX)) :- query_entail(node(X), ex(EX)).


% there is an active support link if source node is activated and no inbound attack to that node.
support(node(X), node(Y), ex(EX)) :- link(support(node(X), node(Y))), node(X, ex(EX)), not attack(node(_), node(X), ex(EX)).


% there is an active attack link if source node is activated and no inbound attack to that node.
attack(node(X), node(Y), ex(EX)) :- link(attack(node(X), node(Y))), node(X, ex(EX)), not attack(node(_), node(X), ex(EX)).

% node is activated
node(Y, ex(EX)) :- support(node(X), node(Y), ex(EX)), node(X, ex(EX)).


% there must be an active support link to pledged node if pledged node is activated
1 <= {support(node(afactor(Y)), node(pledged), ex(EX)): node(afactor(Y), ex(EX)) }:- pledged(true, ex(EX)).



:- pledged(false, ex(EX)), node(pledged, ex(EX)).

% if there are active support link, there must be an active source node.
node(afactor(Y), ex(EX)) :- support(node(afactor(Y)), node(pledged), ex(EX)).

link(support(node(afactor(Y)), node(pledged))):- support(node(afactor(Y)), node(pledged), ex(EX)).

link(attack(node(afactor(Y)), node(afactor(X)))):- attack(node(afactor(Y)), node(afactor(X)), ex(EX)).

% generate link type
1 {link(attack(node(afactor(X)), node(afactor(Y)))): node(afactor(Y)); link(support(node(afactor(X)), node(pledged))): node(pledged)} :- node(afactor(X)).


1 {link(support(node(afactor(X)), node(pledged))): node(afactor(X))} :- node(pledged).  

% constraint on link type
:- link(attack(node(Y), node(pledged))).
:- link(support(node(afactor(Y)), node(afactor(X)))).
:- link(attack(X,X)).
:- link(support(Y,Y)).
:- link(attack(node(afactor(Y)), node(bfactor(Z)))).
:- link(support(node(afactor(Y)), node(bfactor(Z)))).


#minimize { 1@1, L: link(L) }.
% #show node/1.
#show.
#show link(attack(node(afactor(X)), node(afactor(Y)))): link(attack(node(afactor(X)), node(afactor(Y)))).
#show link(support(node(afactor(X)), node(pledged)): link(support(node(afactor(X)), node(pledged))).
#show link(support(node(bfactor(X)), node(afactor(Y)))): link(support(node(bfactor(X)), node(afactor(Y)))).