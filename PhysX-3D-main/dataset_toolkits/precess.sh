
python merge_property.py --datapath /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/   # your physxnet path

python gen_csv.py

python retrieval_texture_example.py

python build_metadata.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output/4368

python render.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output/4368

python build_metadata.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output
python render_cond.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output

python build_metadata.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output

python voxelize.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output
python build_metadata.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output







python extract_feature.py --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output
python build_metadata.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output


python encode_latent.py --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output
python encode_latent_phy.py --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output
python build_metadata.py PhysXNet --output_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output



