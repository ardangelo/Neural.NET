using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;

using Neural;
using Microsoft.FSharp.Core;
using Microsoft.FSharp.Collections;
using MathNet.Numerics.LinearAlgebra;

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=234238

namespace NeuralAppTest {

    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
		Network network;
		List<Matrix<double>> weights;
		List<Vector<double>> biases;

		public MainPage() {
			this.InitializeComponent();

			this.NavigationCacheMode = NavigationCacheMode.Required;

			SumSwitch.IsEnabled = false;
			CarrySwitch.IsEnabled = false;

			var M = Matrix<Double>.Build;
			var V = Vector<Double>.Build;
			weights = new List<Matrix<double>>(){
				M.DenseOfRowArrays(new[] {1.0,0.0}, new[] {-2.0,-2.0}, new[] {0.0,1.0}),
				M.DenseOfRowArrays(new[] {-2.0,-2.0,0.0}, new[] {0.0,-2.0,-2.0}, new[] {0.0,1.0,0.0}),
				M.DenseOfRowArrays(new[] {-2.0,-2.0,0.0}, new[] {0.0,0.0,-4.0}),
			};
			biases = new List<Vector<double>>(){
				V.DenseOfArray(new[] {0.0,3.0,0.0}),
				V.DenseOfArray(new[] {3.0,3.0,0.0}),
				V.DenseOfArray(new[] {3.0,3.0})
			};

			network = new Network(Activations.Step.Activation, Activations.Step.Prime, weights, biases);
		}

        /// <summary>
        /// Invoked when this page is about to be displayed in a Frame.
        /// </summary>
        /// <param name="e">Event data that describes how this page was reached.
        /// This parameter is typically used to configure the page.</param>
        protected override void OnNavigatedTo(NavigationEventArgs e)
        {
            // TODO: Prepare page for display here.

            // TODO: If your application contains multiple pages, ensure that you are
            // handling the hardware Back button by registering for the
            // Windows.Phone.UI.Input.HardwareButtons.BackPressed event.
            // If you are using the NavigationHelper provided by some templates,
            // this event is handled for you.
        }

		private void Switch_Toggled(object sender, RoutedEventArgs e)
		{
			double one = 0;
			double two = 0;
			if (Switch1.IsOn) {
				one = 1.0;
			}
			if (Switch2.IsOn) {
				two = 1.0;
			}

			var a = Vector<double>.Build.DenseOfArray(new[] {one, two});
			var output = network.Output(a);

			double sum = output.At(0);
			double carry = output.At(1);

			if (sum == 0.0) {
				SumSwitch.IsOn = false;
			} else {
				SumSwitch.IsOn = true;
			}
			
			if (carry == 0.0) {
				CarrySwitch.IsOn = false;
			} else {
				CarrySwitch.IsOn = true;
			}
		}
    }
}
