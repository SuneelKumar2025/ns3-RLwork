/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 *
 */

#include "ns3/core-module.h"
#include "ns3/opengym-module.h"
// new moddules
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-mac.h"
#include "ns3/config.h"
#include "ns3/yans-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/mobility-helper.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/udp-client-server-helper.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ht-frame-exchange-manager.h" // Make sure to include the header
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <algorithm>


using namespace ns3;
  double simulationTime = 40; //seconds
NS_LOG_COMPONENT_DEFINE ("OpenGym");


/* * Callback: Tells Python what the network state "looks like".
 * Here: A vector of 5 values (0.0 to 10.0).
 */
Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
  uint32_t nodeNum = 4;
  float low = 0.0;
  float high = 1000000.0; // Increase this to accommodate your large numbers
  std::vector<uint32_t> shape = {nodeNum,};
  std::string dtype = TypeNameGet<float>();
  return CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
}

/* * Callback: Tells Python what the network state "looks like".
 * Here: A vector of 5 values (0.0 to 10.0).
 */
Ptr<OpenGymSpace> MyGetActionSpace(void)
{
  // Size 2 means the agent can choose 0 or 1
  uint32_t actionCount = 2; 
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (actionCount);
  return space;
}

/* * Callback: Determines if simulation should end early.
 */
bool MyGetGameOver(void)
{

  bool isGameOver = false;
  bool test = false;; // Currently hardcoded to false, preventing early exit
  static float stepCounter = 0.0;
  stepCounter += 1;
  if (Simulator::Now().GetSeconds() >= (simulationTime - 0.1)) {
      isGameOver = true;
  }
  // NS_LOG_UNCOND ("MyGetGameOver: " << isGameOver);
  return isGameOver;
}
/* * Callback: Actually generates the network data for Python.
 * Here: Generates 5 random integers to simulate network metrics.
 */
Ptr<OpenGymDataContainer> MyGetObservation(void)
{
  uint32_t nodeNum = 4;
  std::vector<uint32_t> shape = {nodeNum,};
  // Ptr<OpenGymBoxContainer<uint32_t>> box = CreateObject<OpenGymBoxContainer<uint32_t>>(shape);



  Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);
// box->AddValue((float)HtFrameExchangeManager::m_rlState.timestamp);
box->AddValue((float)HtFrameExchangeManager::m_rlState.l1_delay);
box->AddValue((float)HtFrameExchangeManager::m_rlState.l2_delay);
box->AddValue((float)HtFrameExchangeManager::m_rlState.packetSize);
  box->AddValue((float)HtFrameExchangeManager::m_rlState.linkId);
//  NS_LOG_UNCOND (HtFrameExchangeManager::m_rlState.l1_delay<<", "<<HtFrameExchangeManager::m_rlState.l2_delay<<", "<<HtFrameExchangeManager::m_rlState.delay);
  return box;
}
/* * Callback: Actually generates the network data for Python.
 * Here: Generates 5 random integers to simulate network metrics.
 */
// FIX - return a meaningful per-step reward
float MyGetReward(void) {
  // // We use the delay of the link currently being used
  // float currentDelay = HtFrameExchangeManager::m_rlState.delay;
  
  // // Return the negative delay so the agent tries to get as close to 0 as possible
  // // You can scale it so the numbers aren't too large for the Neural Network
  // return - (currentDelay / 100.0); 
  float currentDelay = HtFrameExchangeManager::m_rlState.delay;

  // Override reward: heavy penalty if above threshold, bonus if below
if (currentDelay > 170)
    return  -5   ;                    
else
    return 5   ;                     
}
/*
Define extra info. Optional
*/
std::string MyGetExtraInfo(void)
{
  // Convert the current active delay to a string so Python can read it
  std::string myInfo = std::to_string(HtFrameExchangeManager::m_rlState.delay);
  return myInfo;
}


/* * Callback: Handles the action sent from Python.
 */
bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
{
  // Get the value sent from Python
  Ptr<OpenGymDiscreteContainer> discContainer = DynamicCast<OpenGymDiscreteContainer>(action);
  uint32_t actionVal = discContainer->GetValue();

  // Map 0 to Link 1, and 1 to Link 2
  if (actionVal == 0) {
      // NS_LOG_UNCOND("Switching to Link 1");
      HtFrameExchangeManager::m_rlState.linkId=0;
      // Put your C++ logic here to set the link to 1
  } else if (actionVal == 1) {
      // NS_LOG_UNCOND("Switching to Link 2");
      HtFrameExchangeManager::m_rlState.linkId=1;
      // Put your C++ logic here to set the link to 2
  }

  return true;
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGym)
{
  Simulator::Schedule (Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGym);
  openGym->NotifyCurrentState();
}


//check if we need each espisode random
int
main (int argc, char *argv[])
{
  // Parameters of the scenario
  
  uint32_t simSeed = 1;
  // double simulationTime = 20; //seconds
  double envStepTime = .1; //seconds, ns3gym env step time interval
  uint32_t openGymPort = 5555;
  uint32_t testArg = 0;

  //new parameters
  double distance{1}; //1meters
  int gi = 800;

  std::size_t nStations{3}; //23
  std::string dlAckSeqType{"NO-OFDMA"};
  bool enableUlOfdma{false};
  bool enableBsrp{false};
  int mcs{-1}; // -1 indicates an unset value
  uint32_t payloadSize =   64; // must fit in the max TX duration when transmitting at MCS 0 over an RU of 26 tones

  std::string phyModel{"Yans"};
 

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", simSeed);
  // optional parameters
  cmd.AddValue ("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
  cmd.AddValue ("testArg", "Extra simulation argument. Default: 0", testArg);
  cmd.Parse (argc, argv);

  


  NodeContainer wifiStaNodes;
  wifiStaNodes.Create (nStations);
  NodeContainer wifiApNode;
  wifiApNode.Create (1);

  NetDeviceContainer apDevice, staDevices;
  WifiMacHelper mac;
  WifiHelper wifi1;

  wifi1.SetStandard (WIFI_STANDARD_80211ax);
  Config::SetDefault ("ns3::LogDistancePropagationLossModel::ReferenceLoss", DoubleValue (40));

  std::ostringstream oss1, oss2;
  uint32_t mcs1 = 10; //1pe 24.613 RX 1510510 pe 27.99
  oss1 << "HeMcs" << mcs1;
  

  wifi1.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode",
                                 StringValue (oss1.str ()), "ControlMode",
                                 StringValue (oss1.str ()));
  Ssid ssid = Ssid ("ns3-80211ax");
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy1, phy2;

   // phy1.Set ("Mcs", UintegerValue (7));

  phy1.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
  phy1.SetChannel (channel.Create ());
  phy2.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);
  phy2.SetChannel (channel.Create ());

  mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid));

  phy1.Set ("ChannelSettings", StringValue ("{0 ,160, BAND_6GHZ, 0}")); //phy1 15,160
  phy2.Set ("ChannelSettings", StringValue ("{0, 160, BAND_6GHZ, 0}")); //phy2 47,160

  staDevices = wifi1.Install (phy1, phy2, mac, wifiStaNodes);

  // Disable A-MPDU of station side 1sst
  Ptr<NetDevice> dev;
  Ptr<WifiNetDevice> wifi_dev;
  for (int i = 0; i < nStations; i++)
    {

      dev = wifiStaNodes.Get (i)->GetDevice (0);
      wifi_dev = DynamicCast<WifiNetDevice> (dev);
      wifi_dev->GetMac ()->SetAttribute ("BE_MaxAmpduSize", UintegerValue (0));
      // std::cout<<"TXOP CW:  "<<wifi_dev->GetMac()->GetTxop()->GetMaxCw();///this

    }

  mac.SetType ("ns3::ApWifiMac", "EnableBeaconJitter", BooleanValue (false), "Ssid",
               SsidValue (ssid));
  std::cout << "Scratch:: AP Wifidevice installation\n";
  apDevice = wifi1.Install (phy1, phy2, mac, wifiApNode);
  //Disable A-MPDU
  dev = wifiApNode.Get (0)->GetDevice (0);
  wifi_dev = DynamicCast<WifiNetDevice> (dev);
  wifi_dev->GetMac ()->SetAttribute ("BE_MaxAmpduSize", UintegerValue (0));




  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (simSeed);
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HeConfiguration/GuardInterval",
               TimeValue (NanoSeconds (gi)));
  


  // mobility.
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0)); //AP
  positionAlloc->Add (Vector ((distance), 0.0, 0.0)); //1 MLD
  float gap = 0.1;
  for (int i = 1; i <= nStations; i++, gap += 0.1)
    {
      positionAlloc->Add (Vector (gap, distance, 0.0)); //
    }

  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiApNode);
  mobility.Install (wifiStaNodes);

  /* Internet stack*/
  InternetStackHelper stack;
  stack.Install (wifiApNode);
  stack.Install (wifiStaNodes);
  Ipv4AddressHelper address;
  address.SetBase ("192.168.1.0", "255.255.255.0");
  Ipv4InterfaceContainer staNodeInterfaces;
  Ipv4InterfaceContainer apNodeInterface;

  staNodeInterfaces = address.Assign (staDevices);
  apNodeInterface = address.Assign (apDevice);

    uint16_t port = 98;
// NS_LOG_UNCOND("Initial l1=" << HtFrameExchangeManager::m_rlState.l1_delay
//         << " l2=" << HtFrameExchangeManager::m_rlState.l2_delay
//         << " linkId=" << HtFrameExchangeManager::m_rlState.linkId);


    HtFrameExchangeManager::m_rlState.l1_delay = 0.0;
HtFrameExchangeManager::m_rlState.l2_delay = 0.0;
HtFrameExchangeManager::m_rlState.delay = 0.0;
HtFrameExchangeManager::m_rlState.linkId = 0;
HtFrameExchangeManager::m_rlState.packetSize = 0;
HtFrameExchangeManager::m_rlState.timestamp = 0;
  
 
 for (int i=0;i<=0;i++)
{    
    port++;
    ApplicationContainer serverApp;

    UdpServerHelper server (port);
    serverApp = server.Install (wifiApNode.Get (0));
    serverApp.Start (Seconds (0.0));
    serverApp.Stop (Seconds (simulationTime));

    UdpClientHelper client (apNodeInterface.GetAddress (0), port);
    client.SetAttribute ("MaxPackets", UintegerValue (4294967295u));
    client.SetAttribute ("Interval", TimeValue (Time ("0.001"))); //packets/s 0.000225//0.00112
    client.SetAttribute ("PacketSize", UintegerValue (1500));
    ApplicationContainer clientApp = client.Install (wifiStaNodes.Get (i));

     clientApp.Start (Seconds (.1));
    clientApp.Stop (Seconds (simulationTime));
  
}


// Generate random schedule for STA1 (i=1) and STA2 (i=2)
// Using simSeed so each run gives different order
Ptr<UniformRandomVariable> rng = CreateObject<UniformRandomVariable>();
// rng->SetAttribute("Seed", UintegerValue(simSeed));

// Decide who goes first: if rng > 0.5, STA1 first, else STA2 first
bool sta1First = (rng->GetValue(0.0, 1.0) > 0.5);

// Slots: [0-5], [5-10], [10-15], [15-20]
// STA1 gets slots 0,2 (even) and STA2 gets slots 1,3 (odd) — or vice versa
for (int slot = 0; slot < 4; slot++)
{
    double startTime = slot * 10.0;
    double stopTime  = startTime + 10.0;

    // Who sends in this slot?
    bool sta1Active;
    if (sta1First) {
        sta1Active = (slot % 2 == 0); // STA1 in even slots
    } else {
        sta1Active = (slot % 2 == 1); // STA1 in odd slots
    }

    int activeIdx   = sta1Active ? 1 : 2;
    int inactiveIdx = sta1Active ? 2 : 1;

    port++;
    UdpServerHelper server(port);
    ApplicationContainer serverApp = server.Install(wifiApNode.Get(0));
    serverApp.Start(Seconds(startTime));
    serverApp.Stop(Seconds(stopTime));

    UdpClientHelper client(apNodeInterface.GetAddress(0), port);
    client.SetAttribute("MaxPackets", UintegerValue(4294967295u));
    client.SetAttribute("Interval", TimeValue(Time("0.001")));
    client.SetAttribute("PacketSize", UintegerValue(1500));

    ApplicationContainer clientApp = client.Install(wifiStaNodes.Get(activeIdx));
    clientApp.Start(Seconds(startTime + 0.1));
    clientApp.Stop(Seconds(stopTime));

    // NS_LOG_UNCOND("Slot " << slot << " [" << startTime << "-" << stopTime 
    //               << "]: STA" << activeIdx << " active, STA" << inactiveIdx << " silent");
}

// // =====================================================================
// // NEW SLOT 4 (20.0s to 25.0s): BOTH LINKS CONGESTED SIMULTANEOUSLY
// // =====================================================================
// double finalStart = 10.0;
// double finalStop  = 15.0;

// // 1. Activate STA1 for the final slot
// port++;
// UdpServerHelper serverSta1(port);
// ApplicationContainer serverApp1 = serverSta1.Install(wifiApNode.Get(0));
// serverApp1.Start(Seconds(finalStart));
// serverApp1.Stop(Seconds(finalStop));

// UdpClientHelper clientSta1(apNodeInterface.GetAddress(0), port);
// clientSta1.SetAttribute("MaxPackets", UintegerValue(4294967295u));
// clientSta1.SetAttribute("Interval", TimeValue(Time("0.001"))); 
// clientSta1.SetAttribute("PacketSize", UintegerValue(1500));

// ApplicationContainer clientApp1 = clientSta1.Install(wifiStaNodes.Get(1));
// clientApp1.Start(Seconds(finalStart + 0.1));
// clientApp1.Stop(Seconds(finalStop));

// // 2. Activate STA2 for the exact same slot (forces heavy collision/contention)
// port++;
// UdpServerHelper serverSta2(port);
// ApplicationContainer serverApp2 = serverSta2.Install(wifiApNode.Get(0));
// serverApp2.Start(Seconds(finalStart));
// serverApp2.Stop(Seconds(finalStop));

// UdpClientHelper clientSta2(apNodeInterface.GetAddress(0), port);
// clientSta2.SetAttribute("MaxPackets", UintegerValue(4294967295u));
// clientSta2.SetAttribute("Interval", TimeValue(Time("0.001"))); 
// clientSta2.SetAttribute("PacketSize", UintegerValue(1500));

// ApplicationContainer clientApp2 = clientSta2.Install(wifiStaNodes.Get(2));
// clientApp2.Start(Seconds(finalStart + 0.1));
// clientApp2.Stop(Seconds(finalStop));

  // phy1.EnablePcap ("OpenGym-Phy1-STA", staDevices);
  // phy2.EnablePcap ("OpenGym-Phy2-STA", staDevices);
  // OpenGym Env
  Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface> (openGymPort);
  openGym->SetGetActionSpaceCb( MakeCallback (&MyGetActionSpace) );
  openGym->SetGetObservationSpaceCb( MakeCallback (&MyGetObservationSpace) );
  openGym->SetGetGameOverCb( MakeCallback (&MyGetGameOver) );
  openGym->SetGetObservationCb( MakeCallback (&MyGetObservation) );
  openGym->SetGetRewardCb( MakeCallback (&MyGetReward) );
  openGym->SetGetExtraInfoCb( MakeCallback (&MyGetExtraInfo) );
  openGym->SetExecuteActionsCb( MakeCallback (&MyExecuteActions) );
  Simulator::Schedule (Seconds(0), &ScheduleNextStateRead, envStepTime, openGym);


  // NS_LOG_UNCOND ("Simulation start");
  Simulator::Stop (Seconds (simulationTime));
  Simulator::Run ();
  // NS_LOG_UNCOND ("Simulation stop");

  openGym->NotifySimulationEnd();
  Simulator::Destroy ();

}