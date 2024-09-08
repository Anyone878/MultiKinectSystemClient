#include <string>
#include <chrono>
#include <spdlog/spdlog.h>
#include <k4a/k4a.hpp>
#include <opencv2/opencv.hpp>
#include <csignal>
#include <tuple>

#include <draco/point_cloud/point_cloud.h>
#include <draco/core/decoder_buffer.h>
#include <draco/core/status.h>
#include <draco/metadata/geometry_metadata.h>
#include <draco/compression/decode.h>

#include <shared_mutex>
#include <BS_thread_pool.hpp>
#include <BS_thread_pool_utils.hpp>

#include <enet/enet.h>

#include "../include/Utils/FPSCounter.h"
#include "../include/Utils/MatrixUtils.h"
#include "../include/Utils/ColorChart.h"
#include "../include/Utils/Timer.h"
#include "../include/Utils/ThreadSafeQueue.h"

#include "../include/Cuda/CublasHandleManager.cuh"
#include "../include/Cuda/CudaStreamManager.cuh"
#include "../include/Cuda/CudaTransform.cuh"
#include "../include/Cuda/CudaUtils.cuh"
#include "../include/Cuda/CudaVoxelGridFilter.cuh"

#include "../include/OpenGLFramework.h"
#include "../include/MultiDevice.h"
#include "../include/MultiDeviceTracker.h"
#include "../include/PointCloud.h"

using namespace std;
using namespace cv;
using namespace k4a;

std::atomic<bool> should_stop(false);
constexpr int QUEUE_SIZE = 30;

int main() {
	// cuda streams
	CudaStreamManager& stream_manager = CudaStreamManager::get_instance();
	stream_manager.initialize_streams(1);

	// cublas handle
	CublasHandleManager::get_instance().initialize_handles(1);

	// point cloud decompression (using draco)
	draco::Decoder pc_decoder;

	namedWindow("Test", WINDOW_NORMAL);
	OpenGLFramework app;
	if (!app.init()) {
		spdlog::error("App init failed.");
		return -1;
	}
	vector<OpenGLFramework::PointSet> point_sets;
	vector<OpenGLFramework::JointSet> joint_sets;
	point_sets.emplace_back();

	// init thread pool
	BS::thread_pool pool(std::thread::hardware_concurrency());
	BS::synced_stream sync_out;
	// all results are sync by timestamp
	My::ThreadSafeQueue<std::vector<char>> net2decoder_queue(QUEUE_SIZE);
	My::ThreadSafeQueue<std::tuple<std::vector<My::Point>, std::vector<My::ColorRGB>, std::vector<k4abt_body_t>>> decoder2display_queue(QUEUE_SIZE);

	using namespace std;

	// init client
	if (enet_initialize() != 0) {
		cout << "Init error." << endl;
		return EXIT_FAILURE;
	}
	atexit(enet_deinitialize);

	ENetHost* p_client;
	p_client = enet_host_create(
		NULL,
		5,
		5,
		0, 0
	);
	if (p_client == NULL) {
		cout << "Create client error" << endl;
		exit(EXIT_FAILURE);
	}

	// try to connect
	ENetAddress address;
	enet_address_set_host_ip(&address, "127.0.0.1");
	address.port = 7345;
	ENetPeer* peer = enet_host_connect(p_client, &address, 5, 0);
	if (peer == NULL) {
		cout << "Failed to connect" << endl;
		exit(EXIT_FAILURE);
	}

	pool.detach_task(
		[
			&p_client,
			&peer,
			&net2decoder_queue
		]() {
			ENetEvent event;
			My::Timer timer;
			int iteration_count = 0;
			timer.start();
			while (!should_stop) {
				while (enet_host_service(p_client, &event, 100) > 0) {
					switch (event.type) {
					case ENET_EVENT_TYPE_CONNECT:
						spdlog::info("CONNECT: {}.{}.{}.{}:{}",
							(event.peer->address.host & 0xFF),
							((event.peer->address.host >> 8) & 0xFF),
							((event.peer->address.host >> 16) & 0xFF),
							((event.peer->address.host >> 24) & 0xFF),
							event.peer->address.port);
						peer = event.peer;
						break;
					case ENET_EVENT_TYPE_DISCONNECT:
						spdlog::info("DISCONNECT: {}.{}.{}.{}:{}",
							(event.peer->address.host & 0xFF),
							((event.peer->address.host >> 8) & 0xFF),
							((event.peer->address.host >> 16) & 0xFF),
							((event.peer->address.host >> 24) & 0xFF),
							event.peer->address.port);
						exit(EXIT_SUCCESS);
						break;
					case ENET_EVENT_TYPE_RECEIVE:
						++iteration_count;

						ENetPacket* packet = event.packet;
						spdlog::info("message received, size: {} KB", packet->dataLength / 1024.f);
						vector<char> data(packet->data, packet->data + packet->dataLength);
						net2decoder_queue.push(std::move(data));
						enet_packet_destroy(packet);

						if (iteration_count >= 10) {
							spdlog::info("[DATA RECEIVED] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
							timer.reset();
							timer.start();
							iteration_count = 0;
						}
						break;
					}
				}
			}

		}
	);

	pool.detach_task(
		[
			&net2decoder_queue,
			&decoder2display_queue,
			&pc_decoder
		]() {
			int iteration_count = 0;
			My::Timer timer;

			while (!should_stop) {
				++iteration_count;
				timer.start();

				vector<char> raw_data = net2decoder_queue.pop();
				draco::DecoderBuffer dbuf;
				dbuf.Init(raw_data.data(), raw_data.size());
				draco::StatusOr<std::unique_ptr<draco::PointCloud>> pc_status = pc_decoder.DecodePointCloudFromBuffer(&dbuf);
				if (pc_status.ok()) {
					const draco::PointAttribute* pos_attr = pc_status.value()->GetNamedAttribute(draco::PointAttribute::POSITION);
					const draco::PointAttribute* col_attr = pc_status.value()->GetNamedAttribute(draco::PointAttribute::COLOR);
					std::vector<My::Point> points(pos_attr->size());
					std::copy(
						reinterpret_cast<My::Point*>(pos_attr->buffer()->data()),
						reinterpret_cast<My::Point*>(pos_attr->buffer()->data()) + pos_attr->size(),
						points.begin()
					);

					std::vector<My::ColorRGB> colors(col_attr->size());
					std::copy(
						reinterpret_cast<My::ColorRGB*>(col_attr->buffer()->data()),
						reinterpret_cast<My::ColorRGB*>(col_attr->buffer()->data()) + col_attr->size(),
						colors.begin()
					);

					// std::tuple<std::vector<My::Point>, std::vector<My::ColorRGB>, std::vector<k4abt_body_t>>
					decoder2display_queue.push(std::move(
						std::make_tuple(
							std::move(points), std::move(colors), std::vector<k4abt_body_t>(0)
						)
					));

					draco::GeometryMetadata* geo_meta = pc_status.value()->metadata();
					spdlog::info("num entries: {}", geo_meta->num_entries());
				}
				else {
					spdlog::error("decode error.");
				}

				timer.pause();
				if (iteration_count >= 20) {
					spdlog::info("[DECODER] {} FPS.", 1000 / (timer.elapsed() / iteration_count));
					iteration_count = 0;
					timer.reset();
				}
			}
		}
	);

	// capture and display
	while (!app.windowShouldClose()) {
		// display
		//for (int i = 0; i < body_tracking_output.size(); ++i) {
		//	joint_sets.emplace_back();
		//	for (int j = 0; j < K4ABT_JOINT_COUNT; ++j) {
		//		joint_sets[i].add_joint(
		//			glm::vec3(
		//				body_tracking_output[i].skeleton.joints[j].position.xyz.x,
		//				body_tracking_output[i].skeleton.joints[j].position.xyz.y,
		//				body_tracking_output[i].skeleton.joints[j].position.xyz.z
		//			),
		//			COLOR_CHART[i % COLOR_CHART_COUNT]
		//		);
		//	}
		//}
		std::tuple<std::vector<My::Point>, std::vector<My::ColorRGB>, std::vector<k4abt_body_t>> t = decoder2display_queue.pop();
		std::vector<My::Point>& p = std::get<0>(t);
		std::vector<My::ColorRGB>& c = std::get<1>(t);
		std::vector<My::ColorBGRA> new_c;
		cuda_RGB2BGRA(c, new_c);

		point_sets[0].update(
			reinterpret_cast<int16_t*>(p.data()),
			reinterpret_cast<uint8_t*>(new_c.data()),
			p.size()
		);

		app.beforeUpdate();
		app.updatePointCloud(point_sets);
		app.updateJoints(joint_sets);
		app.afterUpdate();

		for (OpenGLFramework::JointSet& js : joint_sets) {
			js.release_joints();
		}
	}
	should_stop = true;

	// destory
	enet_host_destroy(p_client);

	return 0;
}
