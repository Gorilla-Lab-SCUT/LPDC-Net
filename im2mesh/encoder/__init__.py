from im2mesh.encoder import  pointnet


encoder_dict = {
    'pointnet_simple': pointnet.SimplePointnet,
}

encoder_temporal_dict = {
    'pointnet_spatiotemporal': pointnet.SpatioTemporalResnetPointnet,
    'pointnet_spatiotemporal2': pointnet.SpatioTemporalResnetPointnet2, 
}