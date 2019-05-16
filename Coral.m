function Xs_t = Coral(Xs,Xt)
    cov_source = cov(Xs) + eye(size(Xs, 2));
    cov_target = cov(Xt) + eye(size(Xt, 2));
    A_coral = cov_source^(-1/2)*cov_target^(1/2);
    Xs_t = Xs*A_coral;
end