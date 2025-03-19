from .sp_sg_src import SPSG
from .sp_sg_sparse_seperate import SPSGSparseSeparate
from .sp_sg_src_lite import SPSGLite


def get_model(name, dataset):
    return {
            'sp_sg_attn'        : SPSG,
            'sp_sg_sparse_separate': SPSGSparseSeparate,
            'sp_sg_attn_lite'   : SPSGLite,
           }[name]
