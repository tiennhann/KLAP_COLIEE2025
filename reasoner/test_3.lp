node(root(1)). node(afactor(1;2;3;7)). node(bfactor(1..7)).


% 
link(support(node(bfactor(1)), node(afactor(1)))). 
link(support(node(bfactor(7)), node(afactor(7)))). 
link(support(node(bfactor(4)), node(afactor(2)))). 
link(support(node(bfactor(5)), node(afactor(2)))). 
link(support(node(bfactor(6)), node(afactor(2)))). 
link(support(node(bfactor(2)), node(afactor(3)))). 
link(support(node(bfactor(3)), node(afactor(3)))).

% example 4
query_entail(node(bfactor(1)), ex(4)). query_entail(node(bfactor(2)), ex(4)). root(1, true, ex(4)).

% example 1
query_entail(node(bfactor(1)), ex(1)). query_entail(node(bfactor(7)), ex(1)). root(1, false, ex(1)).

% example 2
query_entail(node(bfactor(3)), ex(2)). query_entail(node(bfactor(7)), ex(2)). root(1, false, ex(2)).

% example 3
query_entail(node(bfactor(1)), ex(3)). root(1, true, ex(3)).


% example 5
query_entail(node(bfactor(1)), ex(5)). query_entail(node(bfactor(2)), ex(5)). query_entail(node(bfactor(7)), ex(5)). root(1, true, ex(5)).

% example 6
query_entail(node(bfactor(1)), ex(6)). query_entail(node(bfactor(2)), ex(6)). query_entail(node(bfactor(6)), ex(6)). query_entail(node(bfactor(7)), ex(6)). root(1, true, ex(6)).


% correct link
% link(support(node(afactor(1)),node(root(1)))). 
% link(attack(node(afactor(7)),node(afactor(1)))). 
% link(attack(node(afactor(2)),node(afactor(1)))). 
% link(attack(node(afactor(3)),node(afactor(7)))). 
% link(attack(node(afactor(3)),node(afactor(2)))).