from pycutfem.utils.bitset import BitSet
def test_bitset():
    a=BitSet([True,False,True])
    b=BitSet([True,True,False])
    assert (a&b).to_indices().tolist()==[0]
    assert (a|b).cardinality()==3
