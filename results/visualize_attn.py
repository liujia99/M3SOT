import numpy as np
import open3d as o3d
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle

# read your data,  including coordinates and features of point cloud
path = 'attn/m3sot_kitti_cyclist_test_multi_input1_perception_space_test/2023-08-14_18-15-25.pkl'
f = open(path, 'rb')
data = pickle.load(f)
# print('data=>', data)
for i in range(len(data)):
    l_search_feat = data[i]['l_search_feat']
    l_search_xyz = data[i]['l_search_xyz']
    l_template_feat = data[i]['l_template_feat']
    l_template_xyz = data[i]['l_template_xyz']
    for j in range(len(l_search_feat)):
        print('l_search_xyz[j]=>', l_search_xyz[j], l_search_xyz[j].shape)
        print('l_search_feat[j]=>', l_search_feat[j], l_search_feat[j].shape)
        embedded_feat = TSNE(n_components=1, learning_rate='auto', init='pca', n_jobs=-1).fit_transform(l_search_feat[j].T)
        print('embedded_feat=>', embedded_feat, embedded_feat.shape)
        feat_min = embedded_feat.min(0)
        feat_max = embedded_feat.max(0)
        norm_embedded_feat = (embedded_feat - feat_min) / (feat_max - feat_min)
        norm_embedded_feat = norm_embedded_feat.reshape(-1)
        colorMap = plt.get_cmap('jet_r')  # plt.get_cmap('jet')
        color = colorMap(norm_embedded_feat)
        color = np.array(color)[:, :3]
        search = o3d.geometry.PointCloud()
        search.points = o3d.utility.Vector3dVector(l_search_xyz[j])
        search.colors = o3d.utility.Vector3dVector(color)

        o3d.io.write_point_cloud("attn/m3sot_kitti_cyclist_test_multi_input1_perception_space_test/tracklet_%d_%d_s.ply" % (i,j), search)
        # o3d.visualization.draw_geometries([search])

        embedded_feat = TSNE(n_components=1, learning_rate='auto', init='pca', n_jobs=-1).fit_transform(l_template_feat[j].T)
        feat_min = embedded_feat.min(0)
        feat_max = embedded_feat.max(0)
        norm_embedded_feat = (embedded_feat - feat_min) / (feat_max - feat_min)
        norm_embedded_feat = norm_embedded_feat.reshape(-1)
        colorMap = plt.get_cmap('jet_r')  # plt.get_cmap('jet')
        color = colorMap(norm_embedded_feat)
        color = np.array(color)[:, :3]
        template = o3d.geometry.PointCloud()
        template.points = o3d.utility.Vector3dVector(l_template_xyz[j])
        template.colors = o3d.utility.Vector3dVector(color)

        o3d.io.write_point_cloud("attn/m3sot_kitti_cyclist_test_multi_input1_perception_space_test/tracklet_%d_%d_t.ply" % (i, j), template)
        # o3d.visualization.draw_geometries([template])