#
# Other functions.
#

function parity(s::Int, i::Int, j::Int)
    c = 0
    for k=min(i, j)+1:max(i, j)-1
        c += (s>>k)&1
    end
    return isodd(c) ? -1. : 1.
end
