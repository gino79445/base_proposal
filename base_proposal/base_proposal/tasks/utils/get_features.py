import torch
import cv2
import numpy as np
from torch.nn.functional import interpolate
import torch.hub
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        # Load DINO-v2 model
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)

    def _get_features(self, transformed_rgb, shape_info):
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        assert img_tensors.shape[1] == 3, "Unexpected image shape, expected 3 color channels."
        features_dict = self.dinov2.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens'].reshape(1, patch_h, patch_w, -1)
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2), size=(720, 1280), mode='bilinear').permute(0, 2, 3, 1).squeeze(0)
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])
        return features_flat

def prepare_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image

def get_3d_point( u, v, Z, R, T, fx, fy, cx, cy):
        # Retrieve camera parameters
        
        # Convert pixel coordinates to normalized camera coordinates
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        # Convert to camera coordinates (as column vector)
        point = np.array([[Z], [X], [Y]])
        # Apply the inverse transformation (R^-1 and translation)
        R_inv = np.linalg.inv(R)
        point_3d = R_inv @ point + T 
        return point_3d


def get_features(R, T, fx, fy, cx, cy, depth_image, K):
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    proposer = KeypointProposer(config)

    image_path = './data/original.png'
    mask_path = './data/target_mask.png'
    original_image = cv2.imread(image_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask_image, 127, 1, cv2.THRESH_BINARY)

    origin_h, origin_w = original_image.shape[:2]
    processed_image = prepare_image(image_path)

    shape_info = {
        'img_h': processed_image.shape[0],
        'img_w': processed_image.shape[1],
        'patch_h': processed_image.shape[0] // 14,
        'patch_w': processed_image.shape[1] // 14,
    }
    features_flat = proposer._get_features(processed_image, shape_info).detach().cpu().numpy()
    normalized_features = normalize(features_flat, norm='l2', axis=1)
    #point = self.get_3d_point(i, j, depth[self.rgb_data.shape[0] - j,self.rgb_data.shape[1] - i], R, T, fx, fy, cx, cy)
    masked_features = normalized_features[mask.flatten().astype(bool)]

    kmeans = KMeans(n_clusters=K, random_state=42)
    cluster_labels = kmeans.fit_predict(masked_features)

    colors = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255], 
        [255, 255, 0], [0, 255, 255], [255, 0, 255], 
        [0, 128, 128], [128, 0, 0], [0, 128, 128], [128, 0, 128]
    ])

    #colors = np.random.randint(0, 255, (10, 3))
    color_mask = np.zeros((origin_h, origin_w, 3), dtype=np.uint8)
    color_mask[mask.astype(bool)] = colors[cluster_labels]

    alpha = 0.1
    #final_image = cv2.addWeighted(original_image, 1 - alpha, color_mask, alpha, 0)
    final_image = original_image.copy()

    # Update to find closest valid point within the mask for each centroid
    cluster_points = np.column_stack(np.where(mask))
    number = 1
    centroid_points = []
    number_list = []
    for i in range(kmeans.n_clusters):
        # Extract all points in the current cluster
        points = cluster_points[cluster_labels == i]
        distance = np.linalg.norm(points - points.mean(axis=0), axis=1)
        closest_point = points[np.argmin(distance)]
        centroid = closest_point

        # Compute the mean (centroid) of these points
        #centroid = points.mean(axis=0).astype(int)
        Continue = False
        for point in centroid_points:
            i, j = point
            P = get_3d_point(origin_w - j, origin_h - i, depth_image[i, j], R, T, fx, fy, cx, cy)
            Q = get_3d_point(origin_w - centroid[1], origin_h - centroid[0], depth_image[centroid[0], centroid[1]], R, T, fx, fy, cx, cy)
            #calculate the distance between the centroid and the point
            distance = np.linalg.norm(P - Q)
            if distance < 0.1:
                Continue = True
                break
        if Continue:
            number += 1
            continue
        centroid_points.append(centroid)
        number_list.append(number)
        #cv2.circle(final_image, (centroid[1], centroid[0]), 10, (255, 255, 255), -1)
        #cv2.circle(final_image, (centroid[1], centroid[0]), 10, (0, 0, 255), 1)
        #text_width, text_height = cv2.getTextSize(f"{number}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        #cv2.putText(final_image, f"{number}", (centroid[1] - text_width // 2, centroid[0] + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y, x = centroid
        #distance = np.linalg.norm(points - centroid, axis=1)
        #closest_point = points[np.argmin(distance)]
        #y, x = closest_point
        cv2.circle(final_image, (x, y), 10, (255, 255, 255), -1)
        cv2.circle(final_image, (x, y), 10, (0, 0, 255), 1)
        text_width, text_height = cv2.getTextSize(f"{number}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(final_image, f"{number}", (x - text_width // 2, y + text_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        number += 1
       
        

       # if len(points) > 0:

       #     # Find the closest point in the cluster to the computed centroid
       #     distances = np.linalg.norm(points - centroid, axis=1)
       #     closest_point = points[np.argmin(distances)]

       #     y, x = closest_point
       #     cv2.circle(final_image, (x, y), 3, (0, 0, 255), -1)
       #     cv2.putText(final_image, f"{number}", (x - 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (170, 22, 219), 2)
       #     number_list.append(number)
       #     number += 1

    cv2.imwrite('./data/clustered_image.png', final_image)
    return cluster_points, cluster_labels , number_list

    
   # # get the mask of the count = 1
   # mask = np.zeros_like(mask)
   # points = cluster_points[cluster_labels == 0]
   # for point in points:
   #     mask[point[0], point[1]] = 255

   # #save the mask
   # cv2.imwrite('./data/mask.png', mask)

#main = get_features
#if __name__ == "__main__":
#    main()

