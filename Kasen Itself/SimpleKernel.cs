using System;
using System.IO;
using System.Threading;
using Cosmos.Core;
using Cosmos.System;

namespace SimpleCosmosKernel
{
    public class Kernel : CosmosKernel
    {
        public static string sharedFilePath = "C:/users/fpsboost/Desktop/SimpleOS_Folder/SharedBridge.txt";

        protected override void Run()
        {
            // Initializing the system
            Console.WriteLine("SimpleCosmosKernel Started!");
            Thread.Sleep(1000); // Allow for initial boot time

            while (true)
            {
                // Send System Info and Time Sync to Python OS
                SendSystemInfo();
                SendTimeSync();

                // Wait for 1 second before sending data again
                Thread.Sleep(1000);
            }
        }

        private void SendSystemInfo()
        {
            string cpuUsage = GetCPUUsage();
            string ramUsage = GetRAMUsage();
            string systemInfo = $"CPU Usage: {cpuUsage} | RAM Usage: {ramUsage}";

            // Writing system info to shared file
            WriteToSharedFile("SYSTEM INFO", systemInfo);
        }

        private string GetCPUUsage()
        {
            // Example of simulating CPU usage info (this would be more complex in a real scenario)
            return "50%"; // Placeholder value
        }

        private string GetRAMUsage()
        {
            // Example of simulating RAM usage info (this would be more complex in a real scenario)
            return "70%"; // Placeholder value
        }

        private void SendTimeSync()
        {
            string timeSync = $"TIME SYNC: {DateTime.UtcNow}";

            // Writing time sync info to shared file
            WriteToSharedFile("TIME SYNC", timeSync);
        }

        private void WriteToSharedFile(string header, string message)
        {
            try
            {
                using (StreamWriter writer = new StreamWriter(sharedFilePath, append: true))
                {
                    writer.WriteLine($"--- {header} ---");
                    writer.WriteLine(message);
                    writer.WriteLine("----------------");
                }
                Console.WriteLine($"Written to {sharedFilePath}: {header} - {message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Failed to write to shared file: {ex.Message}");
            }
        }
    }
}
