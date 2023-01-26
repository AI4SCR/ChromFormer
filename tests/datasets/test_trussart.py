from ChromFormer.datasets.trussart import Trussart


# %%
def test_load_trussart():
    ds = Trussart()
    assert True

def test_trussart_data_equivalence():
    from ChromFormer.Data_Tools.Data_Calculation import import_trussart_data
    trussart_hic_old, trussart_structures_old = import_trussart_data('/Users/adrianomartinelli/projects/ChromFormer/data-old')

    trussart = Trussart()
    trussart_hic, trussart_structures = trussart.data

    assert (trussart_hic == trussart_hic_old).all()

    for new, old in zip(trussart_structures, trussart_structures_old):
        assert (new == old).all()