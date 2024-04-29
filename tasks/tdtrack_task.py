import torch.nn.functional as F
import torch
from datetime import datetime
import pickle as pkl
import os.path as osp

from datasets.utils.pcd_utils import *
from .base_task import BaseTask

class TDTrackTask_Multi_Input_Perception(BaseTask): # Same as TDTrackTask_Multi_Input_Perception_Space

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

    def training_step(self, batch, batch_idx):

        pred = self.model(batch)
        search_bbox_gt = batch['search_bbox_gt']
        l_template_bbox_gt = batch['l_template_bbox_gt'] # b,size,n,3 > [b,n,3 ... b,n,3]
        l_template_bbox_gt = [l_template_bbox_gt[:, i, ...] for i in range(l_template_bbox_gt.shape[1])]

        l_search_mask_gt = pred.pop('l_search_mask_gt')
        l_template_mask_gt = pred.pop('l_template_mask_gt')
        l_center_xyz = pred.pop('l_center_xyz')  # vote xyz
        l_refined_bboxes = pred.pop('l_refined_bboxes')
        # print('pred.keys()=>', pred.keys())
        loss_cascaded_mask = 0
        layer_num = len(self.cfg.model_cfg.transformer_cfg.layers_cfg)
        for k in list(pred.keys()):
            if k.startswith('template_mask_pred_'):
                i = int(k.split('_')[-1])
                t_m_pd = pred.pop('template_mask_pred_%d' % i)
                loss_cascaded_mask += F.binary_cross_entropy_with_logits(
                    t_m_pd,
                    l_template_mask_gt[int(i/layer_num)],
                    pos_weight=torch.tensor([1.0], device=self.device)
                )
            elif k.startswith('search_mask_pred_'):
                i = int(k.split('_')[-1])
                s_m_pd = pred.pop('search_mask_pred_%d' % i)
                search_mask_gt = l_search_mask_gt[-1]
                try:
                    loss_cascaded_mask += F.binary_cross_entropy_with_logits(
                        s_m_pd, l_search_mask_gt[int(i/layer_num)],
                        pos_weight=torch.tensor([1.0], device=self.device)
                    )
                except:
                    loss_cascaded_mask += F.binary_cross_entropy_with_logits(
                        s_m_pd, search_mask_gt,
                        pos_weight=torch.tensor([1.0], device=self.device)
                    )

        if self.cfg.loss_cfg.cascaded_center_loss_func == 'smooth_l1':
            loss_func = F.smooth_l1_loss
        elif self.cfg.loss_cfg.cascaded_center_loss_func == 'mse':
            loss_func = F.mse_loss

        loss_cascaded_center = 0
        for k in list(pred.keys()):
            if k.startswith('template_center_pred_'):
                i = int(k.split('_')[-1])
                t_c_pd = pred.pop('template_center_pred_%d' % i)
                t_c_gt = l_template_bbox_gt[int(i/layer_num)][:, 0:3].unsqueeze(1).expand_as(t_c_pd)
                loss_center = loss_func(t_c_pd, t_c_gt, reduction='none')
                loss_center = (loss_center.mean(2) * l_template_mask_gt[int(i/layer_num)]).sum() \
                              / (l_template_mask_gt[int(i/layer_num)].sum() + 1e-06)
                loss_cascaded_center += loss_center
            elif k.startswith('search_center_pred_'):
                i = int(k.split('_')[-1])
                s_c_pd = pred.pop('search_center_pred_%d' % i)
                s_c_gt = search_bbox_gt[:, 0:3].unsqueeze(1).expand_as(s_c_pd)
                loss_center = loss_func(s_c_pd, s_c_gt, reduction='none')
                search_mask_gt = l_search_mask_gt[-1]
                try:
                    loss_center = (loss_center.mean(2) * l_search_mask_gt[int(i/layer_num)]).sum() \
                                  / (l_search_mask_gt[int(i/layer_num)].sum() + 1e-06)
                except:
                    loss_center = (loss_center.mean(2) * search_mask_gt).sum() / (search_mask_gt.sum() + 1e-06)
                loss_cascaded_center += loss_center

        loss_perceptive_refined_mask, loss_perceptive_refined_box = 0, 0
        for i in range(len(l_center_xyz)):
            dist = torch.norm(l_center_xyz[i] - search_bbox_gt[:, None, :3], p=2, dim=-1)
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_score = l_refined_bboxes[i][:, :, 4]  # B, K
            objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
            objectness_mask[dist < 0.3] = 1
            objectness_mask[dist > 0.6] = 1
            if self.cfg.model_cfg.exrpn_cfg:
                loss_refined_mask = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                                       pos_weight=torch.tensor([2.0], device=self.device))
                loss_refined_mask = torch.sum(loss_refined_mask * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
                loss_perceptive_refined_mask += loss_refined_mask
            elif self.cfg.model_cfg.rpn_cfg:
                loss_refined_mask = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                                       pos_weight=torch.tensor([2.0], device=self.device), reduction='none')
                loss_refined_mask = torch.sum(loss_refined_mask * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
                loss_perceptive_refined_mask += loss_refined_mask

            if self.cfg.loss_cfg.refined_loss_func == 'smooth_l1':
                loss_func = F.smooth_l1_loss
            elif self.cfg.loss_cfg.refined_loss_func == 'mse':
                loss_func = F.mse_loss

            loss_refined_box = loss_func(l_refined_bboxes[i][:, :, :4], search_bbox_gt[:, None, :4].expand_as(l_refined_bboxes[i][:, :, :4]), reduction='none')
            loss_refined_box = torch.sum(loss_refined_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)
            loss_perceptive_refined_box += loss_refined_box

        loss = loss_cascaded_center * self.cfg.loss_cfg.cascaded_center_weight + \
               loss_cascaded_mask * self.cfg.loss_cfg.cascaded_mask_weight + \
               loss_perceptive_refined_box * self.cfg.loss_cfg.refined_box_weight + \
               loss_perceptive_refined_mask * self.cfg.loss_cfg.refined_mask_weight

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_cascaded_center': loss_cascaded_center,
                'loss_perceptive_refined_mask': loss_perceptive_refined_mask,
                'loss_perceptive_refined_box': loss_perceptive_refined_box,
                'loss_cascaded_mask': loss_cascaded_mask,
            },
            global_step=self.global_step
        )

        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(
                v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data

    def forward_on_tracklet(self, tracklet):
        pred_bboxes = []
        gt_bboxes = []
        first_frame = tracklet[0]
        data = []
        for frame_id, frame in enumerate(tracklet):
            if frame_id == 0:
                pred_bboxes.append(frame['bbox'])
                gt_bboxes.append(frame['bbox'])
                continue
            if self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'previous_pred':
                ref_bbox = pred_bboxes[-1]
            elif self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'previous_gt':
                ref_bbox = gt_bboxes[-1]
            elif self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'current_gt':
                ref_bbox = frame['bbox']

            search_pcd = crop_and_center_pcd(
                frame['pcd'], ref_bbox, offset=self.cfg.dataset_cfg.search_offset, offset2=self.cfg.dataset_cfg.search_offset2, scale=self.cfg.dataset_cfg.search_scale)
            search_pcd = resample_pcd(search_pcd, self.cfg.dataset_cfg.search_npts, is_training=False)
            search_mask_ref = np.ones([search_pcd.points.shape[1], ]) * 0.5
            search_bc_ref = np.zeros([search_pcd.points.shape[1], 9])

            batch_search = dict(
                search_pcd=search_pcd.points.T,
                search_bc_ref=search_bc_ref,
                search_mask_ref=search_mask_ref
            )
            batch_templates = {
                'l_template_pcd': [],
                'l_template_mask_ref': [],
                'l_template_bc_ref': []
            }
            for front in range(self.cfg.dataset_cfg.template_set_size):
                pre_frame_id = max(frame_id - front - 1, 0)
                template_pcd, template_bbox_ref = crop_and_center_pcd(tracklet[pre_frame_id]['pcd'], pred_bboxes[pre_frame_id],
                                          offset=self.cfg.dataset_cfg.template_offset, offset2=self.cfg.dataset_cfg.template_offset2,
                                          scale=self.cfg.dataset_cfg.template_scale, return_box=True)
                template_pcd = resample_pcd(template_pcd, self.cfg.dataset_cfg.template_npts, is_training=False)
                template_bc_ref = get_point_to_box_distance(template_pcd, template_bbox_ref)
                template_mask_ref = get_pcd_in_box_mask(template_pcd, template_bbox_ref, scale=1.25).astype('float32')

                if frame_id != 1:
                    template_mask_ref[template_mask_ref == 0] = 0.2
                    template_mask_ref[template_mask_ref == 1] = 0.8

                batch_templates['l_template_pcd'].append(template_pcd.points.T)
                batch_templates['l_template_mask_ref'].append(template_mask_ref)
                batch_templates['l_template_bc_ref'].append(template_bc_ref)

            batch_templates.update(batch_search)
            pred = self.model(self._to_float_tensor(batch_templates))

            # print('test for visual)
            if not self.training:
                l_search_xyz = [search_xyz.squeeze(0).detach().cpu().numpy() for search_xyz in pred['lv_search_xyz']]
                l_template_xyz = [template_xyz.squeeze(0).detach().cpu().numpy() for template_xyz in pred['lv_template_xyz']]
                l_search_feat = [search_feat.squeeze(0).detach().cpu().numpy() for search_feat in pred['lv_search_feat']]
                l_template_feat = [template_feat.squeeze(0).detach().cpu().numpy() for template_feat in pred['lv_template_feat']]
                data.append(dict(
                    l_search_feat=l_search_feat,
                    l_search_xyz=l_search_xyz,
                    l_template_feat=l_template_feat,
                    l_template_xyz=l_template_xyz,
                ))

            # print('================================')
            estimation_bbox = pred['l_refined_bboxes'][-1]
            for i in range(len(pred['l_refined_bboxes'])-1):
                estimation_bbox = torch.cat([estimation_bbox,pred['l_refined_bboxes'][i]], dim = 1)
            estimation_bbox[:, :, 4] = torch.sigmoid(estimation_bbox[:, :, 4])
            estimation_bbox_cpu = estimation_bbox.squeeze(0).detach().cpu().numpy()
            estimation_bbox_cpu[np.isnan(estimation_bbox_cpu)] = -1e6
            best_box_idx = estimation_bbox_cpu[:, 4].argmax()
            estimation_bbox_cpu = estimation_bbox_cpu[best_box_idx, 0:4]

            bbox = get_offset_box(ref_bbox, estimation_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
            pred_bboxes.append(bbox)
            gt_bboxes.append(frame['bbox'])

        if not self.training:
            with open(osp.join(self.cfg.work_dir, datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')+'.pkl'), 'wb') as f:
                pkl.dump(data, f)
        return pred_bboxes, gt_bboxes

class TDTrackTask_Multi_Input_Perception_Gen(BaseTask):

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

    def training_step(self, batch, batch_idx):

        pred = self.model(batch)
        search_bbox_gt = batch['search_bbox_gt']
        l_template_bbox_gt = batch['l_template_bbox_gt'] # b,size,n,3 > [b,n,3 ... b,n,3]
        l_template_bbox_gt = [l_template_bbox_gt[:, i, ...] for i in range(l_template_bbox_gt.shape[1])]

        search_mask_gt = pred.pop('search_mask_gt')
        l_template_mask_gt = pred.pop('l_template_mask_gt')
        center_xyz = pred.pop('center_xyz')  # vote xyz
        refined_bboxes = pred.pop('refined_bboxes')

        loss_cascaded_mask = 0
        layer_num = len(self.cfg.model_cfg.transformer_cfg.layers_cfg)
        for k in list(pred.keys()):
            if k.startswith('template_mask_pred_'):
                i = int(k.split('_')[-1])
                t_m_pd = pred.pop('template_mask_pred_%d' % i)
                loss_cascaded_mask += F.binary_cross_entropy_with_logits(
                    t_m_pd,
                    l_template_mask_gt[int(i/layer_num)],
                    pos_weight=torch.tensor([1.0], device=self.device)
                )
            elif k.startswith('search_mask_pred_'):
                i = int(k.split('_')[-1])
                s_m_pd = pred.pop('search_mask_pred_%d' % i)
                try:
                    s_m_gt = l_template_mask_gt[int(i/layer_num) + 1]
                except:
                    s_m_gt = search_mask_gt
                loss_cascaded_mask += F.binary_cross_entropy_with_logits(
                    s_m_pd, s_m_gt,
                    pos_weight=torch.tensor([1.0], device=self.device)
                )

        if self.cfg.loss_cfg.cascaded_center_loss_func == 'smooth_l1':
            loss_func = F.smooth_l1_loss
        elif self.cfg.loss_cfg.cascaded_center_loss_func == 'mse':
            loss_func = F.mse_loss

        loss_cascaded_center = 0
        for k in list(pred.keys()):
            if k.startswith('template_center_pred_'):
                i = int(k.split('_')[-1])
                t_c_pd = pred.pop('template_center_pred_%d' % i)
                t_c_gt = l_template_bbox_gt[int(i/layer_num)][:, 0:3].unsqueeze(1).expand_as(t_c_pd)
                loss_center = loss_func(t_c_pd, t_c_gt, reduction='none')
                loss_center = (loss_center.mean(2) * l_template_mask_gt[int(i/layer_num)]).sum() \
                              / (l_template_mask_gt[int(i/layer_num)].sum() + 1e-06)
                loss_cascaded_center += loss_center
            elif k.startswith('search_center_pred_'):
                i = int(k.split('_')[-1])
                s_c_pd = pred.pop('search_center_pred_%d' % i)
                try:
                    s_c_gt = l_template_bbox_gt[int(i/layer_num) + 1][:, 0:3].unsqueeze(1).expand_as(s_c_pd)
                except:
                    s_c_gt = search_bbox_gt[:, 0:3].unsqueeze(1).expand_as(s_c_pd)
                loss_center = loss_func(s_c_pd, s_c_gt, reduction='none')
                loss_center = (loss_center.mean(2) * search_mask_gt).sum() / (search_mask_gt.sum() + 1e-06)
                loss_cascaded_center += loss_center

        loss_perceptive_refined_mask, loss_perceptive_refined_box = 0, 0

        dist = torch.norm(center_xyz - search_bbox_gt[:, None, :3], p=2, dim=-1)
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        objectness_label[dist < 0.3] = 1
        objectness_score = refined_bboxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        if self.cfg.model_cfg.exrpn_cfg:
            loss_refined_mask = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                                   pos_weight=torch.tensor([2.0], device=self.device))
            loss_refined_mask = torch.sum(loss_refined_mask * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
            loss_perceptive_refined_mask += loss_refined_mask
        elif self.cfg.model_cfg.rpn_cfg:
            loss_refined_mask = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                                   pos_weight=torch.tensor([2.0], device=self.device), reduction='none')
            loss_refined_mask = torch.sum(loss_refined_mask * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
            loss_perceptive_refined_mask += loss_refined_mask

        if self.cfg.loss_cfg.refined_loss_func == 'smooth_l1':
            loss_func = F.smooth_l1_loss
        elif self.cfg.loss_cfg.refined_loss_func == 'mse':
            loss_func = F.mse_loss

        loss_refined_box = loss_func(refined_bboxes[:, :, :4], search_bbox_gt[:, None, :4].expand_as(refined_bboxes[:, :, :4]), reduction='none')
        loss_refined_box = torch.sum(loss_refined_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)
        loss_perceptive_refined_box += loss_refined_box

        loss = loss_cascaded_center * self.cfg.loss_cfg.cascaded_center_weight + \
               loss_cascaded_mask * self.cfg.loss_cfg.cascaded_mask_weight + \
               loss_perceptive_refined_box * self.cfg.loss_cfg.refined_box_weight + \
               loss_perceptive_refined_mask * self.cfg.loss_cfg.refined_mask_weight

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_cascaded_center': loss_cascaded_center,
                'loss_perceptive_refined_mask': loss_perceptive_refined_mask,
                'loss_perceptive_refined_box': loss_perceptive_refined_box,
                'loss_cascaded_mask': loss_cascaded_mask,
            },
            global_step=self.global_step
        )

        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(
                v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data

    def forward_on_tracklet(self, tracklet):
        pred_bboxes = []
        gt_bboxes = []
        first_frame = tracklet[0]

        for frame_id, frame in enumerate(tracklet):
            if frame_id == 0:
                pred_bboxes.append(frame['bbox'])
                gt_bboxes.append(frame['bbox'])
                continue
            if self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'previous_pred':
                ref_bbox = pred_bboxes[-1]
            elif self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'previous_gt':
                ref_bbox = gt_bboxes[-1]
            elif self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'current_gt':
                ref_bbox = frame['bbox']

            search_pcd = crop_and_center_pcd(
                frame['pcd'], ref_bbox, offset=self.cfg.dataset_cfg.search_offset, offset2=self.cfg.dataset_cfg.search_offset2, scale=self.cfg.dataset_cfg.search_scale)
            search_pcd = resample_pcd(search_pcd, self.cfg.dataset_cfg.search_npts, is_training=False)
            search_mask_ref = np.ones([search_pcd.points.shape[1], ]) * 0.5
            search_bc_ref = np.zeros([search_pcd.points.shape[1], 9])

            batch_search = dict(
                search_pcd=search_pcd.points.T,
                search_bc_ref=search_bc_ref,
                search_mask_ref=search_mask_ref
            )

            batch_templates = {
                'l_template_pcd': [],
                'l_template_mask_ref': [],
                'l_template_bc_ref': []
            }
            for front in range(self.cfg.dataset_cfg.template_set_size):
                pre_frame_id = max(frame_id - front - 1, 0)
                template_pcd, template_bbox_ref = crop_and_center_pcd(tracklet[pre_frame_id]['pcd'], pred_bboxes[pre_frame_id],
                                          offset=self.cfg.dataset_cfg.template_offset, offset2=self.cfg.dataset_cfg.template_offset2,
                                          scale=self.cfg.dataset_cfg.template_scale, return_box=True)
                template_pcd = resample_pcd(template_pcd, self.cfg.dataset_cfg.template_npts, is_training=False)
                template_bc_ref = get_point_to_box_distance(template_pcd, template_bbox_ref)
                template_mask_ref = get_pcd_in_box_mask(template_pcd, template_bbox_ref, scale=1.25).astype('float32')

                if frame_id != 1:
                    template_mask_ref[template_mask_ref == 0] = 0.2
                    template_mask_ref[template_mask_ref == 1] = 0.8

                batch_templates['l_template_pcd'].append(template_pcd.points.T)
                batch_templates['l_template_mask_ref'].append(template_mask_ref)
                batch_templates['l_template_bc_ref'].append(template_bc_ref)

            batch_templates.update(batch_search)
            pred = self.model(self._to_float_tensor(batch_templates))

            # print('================================')

            estimation_bbox = pred['refined_bboxes']
            estimation_bbox[:, :, 4] = torch.sigmoid(estimation_bbox[:, :, 4])
            estimation_bbox_cpu = estimation_bbox.squeeze(0).detach().cpu().numpy()

            estimation_bbox_cpu[np.isnan(estimation_bbox_cpu)] = -1e6

            best_box_idx = estimation_bbox_cpu[:, 4].argmax()
            estimation_bbox_cpu = estimation_bbox_cpu[best_box_idx, 0:4]

            bbox = get_offset_box(ref_bbox, estimation_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
            pred_bboxes.append(bbox)
            gt_bboxes.append(frame['bbox'])

        return pred_bboxes, gt_bboxes

class TDTrackTask_Multi_Input_Perception_Cross(BaseTask):

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

    def training_step(self, batch, batch_idx):

        pred = self.model(batch)
        search_bbox_gt = batch['search_bbox_gt']
        l_template_bbox_gt = batch['l_template_bbox_gt'] # b,size,n,3 > [b,n,3 ... b,n,3]
        l_template_bbox_gt = [l_template_bbox_gt[:, i, ...] for i in range(l_template_bbox_gt.shape[1])]

        search_mask_gt = pred.pop('search_mask_gt')
        l_template_mask_gt = pred.pop('l_template_mask_gt')
        center_xyz = pred.pop('center_xyz')  # vote xyz
        refined_bboxes = pred.pop('refined_bboxes')

        loss_cascaded_mask = 0
        layer_num = len(self.cfg.model_cfg.transformer_cfg.layers_cfg)
        for k in list(pred.keys()):
            if k.startswith('search_mask_pred_'):
                i = int(k.split('_')[-1])
                s_m_pd = pred.pop('search_mask_pred_%d' % i)
                try:
                    s_m_gt = l_template_mask_gt[int(i/layer_num) + 1]
                except:
                    s_m_gt = search_mask_gt
                loss_cascaded_mask += F.binary_cross_entropy_with_logits(
                    s_m_pd, s_m_gt,
                    pos_weight=torch.tensor([1.0], device=self.device)
                )

        if self.cfg.loss_cfg.cascaded_center_loss_func == 'smooth_l1':
            loss_func = F.smooth_l1_loss
        elif self.cfg.loss_cfg.cascaded_center_loss_func == 'mse':
            loss_func = F.mse_loss

        loss_cascaded_center = 0
        for k in list(pred.keys()):
            if k.startswith('search_center_pred_'):
                i = int(k.split('_')[-1])
                s_c_pd = pred.pop('search_center_pred_%d' % i)
                try:
                    s_c_gt = l_template_bbox_gt[int(i/layer_num) + 1][:, 0:3].unsqueeze(1).expand_as(s_c_pd)
                except:
                    s_c_gt = search_bbox_gt[:, 0:3].unsqueeze(1).expand_as(s_c_pd)
                loss_center = loss_func(s_c_pd, s_c_gt, reduction='none')
                loss_center = (loss_center.mean(2) * search_mask_gt).sum() / (search_mask_gt.sum() + 1e-06)
                loss_cascaded_center += loss_center

        loss_perceptive_refined_mask, loss_perceptive_refined_box = 0, 0

        dist = torch.norm(center_xyz - search_bbox_gt[:, None, :3], p=2, dim=-1)
        objectness_label = torch.zeros_like(dist, dtype=torch.float)
        objectness_label[dist < 0.3] = 1
        objectness_score = refined_bboxes[:, :, 4]  # B, K
        objectness_mask = torch.zeros_like(objectness_label, dtype=torch.float)
        objectness_mask[dist < 0.3] = 1
        objectness_mask[dist > 0.6] = 1
        if self.cfg.model_cfg.exrpn_cfg:
            loss_refined_mask = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                                   pos_weight=torch.tensor([2.0], device=self.device))
            loss_refined_mask = torch.sum(loss_refined_mask * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
            loss_perceptive_refined_mask += loss_refined_mask
        elif self.cfg.model_cfg.rpn_cfg:
            loss_refined_mask = F.binary_cross_entropy_with_logits(objectness_score, objectness_label,
                                                                   pos_weight=torch.tensor([2.0], device=self.device), reduction='none')
            loss_refined_mask = torch.sum(loss_refined_mask * objectness_mask) / (torch.sum(objectness_mask) + 1e-6)
            loss_perceptive_refined_mask += loss_refined_mask

        if self.cfg.loss_cfg.refined_loss_func == 'smooth_l1':
            loss_func = F.smooth_l1_loss
        elif self.cfg.loss_cfg.refined_loss_func == 'mse':
            loss_func = F.mse_loss

        loss_refined_box = loss_func(refined_bboxes[:, :, :4], search_bbox_gt[:, None, :4].expand_as(refined_bboxes[:, :, :4]), reduction='none')
        loss_refined_box = torch.sum(loss_refined_box.mean(2) * objectness_label) / (objectness_label.sum() + 1e-6)
        loss_perceptive_refined_box += loss_refined_box

        loss = loss_cascaded_center * self.cfg.loss_cfg.cascaded_center_weight + \
               loss_cascaded_mask * self.cfg.loss_cfg.cascaded_mask_weight + \
               loss_perceptive_refined_box * self.cfg.loss_cfg.refined_box_weight + \
               loss_perceptive_refined_mask * self.cfg.loss_cfg.refined_mask_weight

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_cascaded_center': loss_cascaded_center,
                'loss_perceptive_refined_mask': loss_perceptive_refined_mask,
                'loss_perceptive_refined_box': loss_perceptive_refined_box,
                'loss_cascaded_mask': loss_cascaded_mask,
            },
            global_step=self.global_step
        )

        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(
                v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data

    def forward_on_tracklet(self, tracklet):
        pred_bboxes = []
        gt_bboxes = []
        first_frame = tracklet[0]

        for frame_id, frame in enumerate(tracklet):
            if frame_id == 0:
                pred_bboxes.append(frame['bbox'])
                gt_bboxes.append(frame['bbox'])
                continue
            if self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'previous_pred':
                ref_bbox = pred_bboxes[-1]
            elif self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'previous_gt':
                ref_bbox = gt_bboxes[-1]
            elif self.cfg.dataset_cfg.eval_cfg.reference_bbox == 'current_gt':
                ref_bbox = frame['bbox']

            search_pcd = crop_and_center_pcd(
                frame['pcd'], ref_bbox, offset=self.cfg.dataset_cfg.search_offset, offset2=self.cfg.dataset_cfg.search_offset2, scale=self.cfg.dataset_cfg.search_scale)
            search_pcd = resample_pcd(search_pcd, self.cfg.dataset_cfg.search_npts, is_training=False)
            search_mask_ref = np.ones([search_pcd.points.shape[1], ]) * 0.5
            search_bc_ref = np.zeros([search_pcd.points.shape[1], 9])

            batch_search = dict(
                search_pcd=search_pcd.points.T,
                search_bc_ref=search_bc_ref,
                search_mask_ref=search_mask_ref
            )

            batch_templates = {
                'l_template_pcd': [],
                'l_template_mask_ref': [],
                'l_template_bc_ref': []
            }
            for front in range(self.cfg.dataset_cfg.template_set_size):
                pre_frame_id = max(frame_id - front - 1, 0)
                template_pcd, template_bbox_ref = crop_and_center_pcd(tracklet[pre_frame_id]['pcd'], pred_bboxes[pre_frame_id],
                                          offset=self.cfg.dataset_cfg.template_offset, offset2=self.cfg.dataset_cfg.template_offset2,
                                          scale=self.cfg.dataset_cfg.template_scale, return_box=True)
                template_pcd = resample_pcd(template_pcd, self.cfg.dataset_cfg.template_npts, is_training=False)
                template_bc_ref = get_point_to_box_distance(template_pcd, template_bbox_ref)
                template_mask_ref = get_pcd_in_box_mask(template_pcd, template_bbox_ref, scale=1.25).astype('float32')

                if frame_id != 1:
                    template_mask_ref[template_mask_ref == 0] = 0.2
                    template_mask_ref[template_mask_ref == 1] = 0.8

                batch_templates['l_template_pcd'].append(template_pcd.points.T)
                batch_templates['l_template_mask_ref'].append(template_mask_ref)
                batch_templates['l_template_bc_ref'].append(template_bc_ref)

            batch_templates.update(batch_search)
            pred = self.model(self._to_float_tensor(batch_templates))

            # print('================================')

            estimation_bbox = pred['refined_bboxes']
            estimation_bbox[:, :, 4] = torch.sigmoid(estimation_bbox[:, :, 4])
            estimation_bbox_cpu = estimation_bbox.squeeze(0).detach().cpu().numpy()

            estimation_bbox_cpu[np.isnan(estimation_bbox_cpu)] = -1e6

            best_box_idx = estimation_bbox_cpu[:, 4].argmax()
            estimation_bbox_cpu = estimation_bbox_cpu[best_box_idx, 0:4]

            bbox = get_offset_box(ref_bbox, estimation_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
            pred_bboxes.append(bbox)
            gt_bboxes.append(frame['bbox'])

        return pred_bboxes, gt_bboxes
