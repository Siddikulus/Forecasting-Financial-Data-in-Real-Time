from sklearn.decomposition import PCA
import pandas as pd

def principalcomponentanalysis(df, target_variable):
    if target_variable in df.columns:
        df = df.drop([target_variable], axis = 1)

    pca = PCA(n_components=0.95, svd_solver='full')
    principal_components = pca.fit_transform(df.values)
    # print(pca.components_)
    principal_df = pd.DataFrame(principal_components, index=df.index, columns = ['Principal_Component_'+ str(i+1) for i in range(principal_components.shape[1])])
    return principal_df