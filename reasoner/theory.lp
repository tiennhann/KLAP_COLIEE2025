% nodes: node(N).
% abstract factors: afactor(N).
% factors: factor(N).
% links: link(Src, Rel, Dst)

% every factor, abstract factor, or root is a node.
% node(N) :- factor(N).
% node(N) :- afactor(N).
% node(N) :- root(N).

% link types
% ltype(0). % support
% ltype(1). % attack

% query link activates bfactor
bfactor(X) :- link(sup(query, bfactor(X))).


afactor(X) :- link(sup(bfactor(Y), afactor(X))), bfactor(Y).



% Rel must be a link type
:- link(_, Rel, _), not ltype(Rel).

% a node is true if its link is satisfied
sat(N) :- node(N), link(S, 0, N), sat(S).
